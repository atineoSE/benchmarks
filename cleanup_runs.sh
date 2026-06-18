#!/bin/bash
# Cleanup utility for incomplete / stuck benchmark runs.
#
# A wedged `*-infer` run (e.g. a HuggingFace download that hangs, or the
# worker-context progress-bar RecursionError) leaves behind state that makes the
# NEXT run get stuck too. Running this with no arguments does the FULL cleanup:
#
#   1. kill matching runner processes (default pattern: "-infer")
#   2. reap their leftover zombies
#   3. delete stale HuggingFace *.lock files
#   4. delete *.incomplete partial downloads (broken, not real downloads)
#   5. remove orphaned OpenHands `--workspace docker` containers
#
# DOWNLOADS ARE PRESERVED: completed HF blobs (hub/) and built dataset caches
# (datasets/) are never touched, so a relaunch re-uses them instead of
# re-downloading. Only broken partials (*.incomplete) and locks are removed.
# The inference stack (see PROTECTED) is never touched.
#
# Usage:  ./cleanup_runs.sh [-n] [-y] [-p PATTERN]
#   -n, --dry-run          show what would happen; change nothing
#   -y, --yes              don't prompt before killing processes
#   -p, --pattern PATTERN  process match (default: "-infer")
#   -h, --help
#
# Safe to re-run. Operates on the HOST HuggingFace cache (HF_HOME, default
# ~/.cache/huggingface) — NOT the vLLM container's model cache.

set -euo pipefail

PATTERN='-infer'
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
DRY_RUN=0
ASSUME_YES=0
# Inference-stack containers that must NEVER be removed.
PROTECTED='vllm-stack|vllm-ingress|nvidia-obs'

GREEN='\033[32m'; YELLOW='\033[33m'; RED='\033[31m'; CYAN='\033[36m'; RESET='\033[0m'
say()  { printf '%b\n' "$*"; }
step() { printf '%b\n' "${CYAN}==>${RESET} $*"; }
usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; s/^set -euo.*//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--dry-run) DRY_RUN=1 ;;
        -y|--yes)     ASSUME_YES=1 ;;
        -p|--pattern) PATTERN="${2:?pattern}"; shift ;;
        -h|--help)    usage; exit 0 ;;
        *) say "${RED}Unknown arg: $1${RESET}"; usage; exit 1 ;;
    esac
    shift
done

(( DRY_RUN )) && say "${YELLOW}[dry-run] no changes will be made${RESET}"

# run CMD, or just echo it under --dry-run
run() { if (( DRY_RUN )); then say "  ${YELLOW}[dry-run]${RESET} $*"; else eval "$@"; fi; }

# --- 1. Kill matching runner processes -------------------------------------
step "Looking for benchmark runners matching: ${CYAN}${PATTERN}${RESET}"
mapfile -t PIDS < <(pgrep -f -- "$PATTERN" | grep -vx "$$" || true)
PIDS=($(for p in "${PIDS[@]:-}"; do
    [[ -z "$p" ]] && continue
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || true)
    [[ "$cmd" == *cleanup_runs* ]] && continue
    echo "$p"
done))

if [[ ${#PIDS[@]} -eq 0 ]]; then
    say "  ${GREEN}none running.${RESET}"
else
    ps -o pid,ppid,etime,stat,cmd -p "$(IFS=,; echo "${PIDS[*]}")" 2>/dev/null | sed 's/^/  /' || true
    if (( ! DRY_RUN && ! ASSUME_YES )); then
        read -rp "$(printf "%bKill these %d process(es)? [y/N] %b" "$YELLOW" "${#PIDS[@]}" "$RESET")" ans
        [[ "$ans" =~ ^[Yy]$ ]] || { say "${RED}Aborted.${RESET}"; exit 1; }
    fi
    run "kill -TERM ${PIDS[*]} 2>/dev/null || true"
    (( DRY_RUN )) || sleep 3
    run "kill -KILL ${PIDS[*]} 2>/dev/null || true"
    say "  ${GREEN}signalled ${#PIDS[@]} process(es).${RESET}"
fi

# --- 2. Reap leftover zombies (kill their parent so init can reap) ----------
step "Reaping defunct (zombie) runners"
ZP=$(ps -eo ppid,stat,comm 2>/dev/null | awk '$2 ~ /Z/ && $3 ~ /(infer|python)/ {print $1}' | sort -u || true)
if [[ -n "$ZP" ]]; then
    for ppid in $ZP; do
        pcmd=$(tr '\0' ' ' < "/proc/$ppid/cmdline" 2>/dev/null || true)
        [[ "$pcmd" == *"$PATTERN"* || "$pcmd" == *uv* ]] && run "kill -TERM $ppid 2>/dev/null || true"
    done
    say "  ${GREEN}poked zombie parents.${RESET}"
else
    say "  ${GREEN}no zombies.${RESET}"
fi

# --- 3+4. Stale HF locks + partial downloads (completed downloads preserved) -
step "Clearing stale HF locks / partial downloads under ${HF_CACHE}"
say "  ${YELLOW}(completed blobs + built datasets are preserved)${RESET}"
if [[ -d "$HF_CACHE" ]]; then
    nlock=$(find "$HF_CACHE" -name '*.lock' 2>/dev/null | wc -l)
    ninc=$(find "$HF_CACHE" -name '*.incomplete' 2>/dev/null | wc -l)
    say "  found ${nlock} lock file(s), ${ninc} incomplete partial(s)"
    run "find '$HF_CACHE' -name '*.lock' -delete 2>/dev/null || true"
    run "find '$HF_CACHE' -name '*.incomplete' -delete 2>/dev/null || true"
    say "  ${GREEN}cleared.${RESET}"
else
    say "  ${YELLOW}HF cache dir not found; skipping.${RESET}"
fi

# --- 5. Remove orphaned OpenHands workspace containers ----------------------
step "Removing orphaned workspace containers (protecting: ${PROTECTED})"
if command -v docker >/dev/null; then
    mapfile -t WS < <(docker ps -a --format '{{.Names}}\t{{.Image}}' 2>/dev/null \
        | grep -iE 'openhands|all-hands|commit0|swebench' \
        | grep -ivE "$PROTECTED" | cut -f1 || true)
    if [[ ${#WS[@]} -eq 0 || -z "${WS[0]:-}" ]]; then
        say "  ${GREEN}none found.${RESET}"
    else
        printf '  %s\n' "${WS[@]}"
        run "docker rm -f ${WS[*]} 2>/dev/null || true"
        say "  ${GREEN}removed ${#WS[@]} container(s).${RESET}"
    fi
else
    say "  ${YELLOW}docker not available; skipping.${RESET}"
fi

say "${GREEN}Cleanup complete${RESET} (downloads preserved). Relaunch with progress bars + Xet disabled:"
say "  ${CYAN}export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1${RESET}"
