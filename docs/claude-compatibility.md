# Open-source LLMs and the Claude Code agent loop: a compatibility evaluation

> **TL;DR.** Claude-Code compatibility is a *training* property, not a
> serving-stack property. Models trained explicitly on agentic curricula
> with RL on tool use — GLM-4.6/4.7, Kimi K2.6, DeepSeek V3.2 — are the
> only ones with concrete evidence of surviving the `acp-claude` loop
> end-to-end. Models trained only for one-shot tool use (base Qwen3,
> Llama, Mistral, Gemma) reliably fail the same way base Qwen3.5 did.

This doc is a companion to
[claude-and-qwen3.5.md](./claude-and-qwen3.5.md), which documents the
exact failure mode the `acp-claude` driver exhibits when paired with
base Qwen3.5. That doc covers *why* a turn ends; this one covers
*which other open-source models would not end it prematurely* if you
swapped them in.

Hardware fit is intentionally out of scope: scaling up the node is on
the table, so the question reduces to behavioural fit with the Claude
Code agent loop.

## What "Claude-Code-compatible" actually tests

The chardet failure pinned down five behaviours a backend must get
right for `acp-claude` to not terminate sessions early:

1. **Mid-task tool-call discipline.** Never finish a generation with
   text-only unless the task is actually done. Every "let me now look
   at X" sentence must come *with* the tool call in the same response.
2. **Reasoning that doesn't swallow tool calls.** If the model uses a
   `<think>`-style channel, the tool call still has to appear in the
   visible output channel — not get embedded in the reasoning trace
   where the parser strips it. (This is the specific Qwen-class
   regression the OpenHands default agent nudges around at
   `agent.py:651`.)
3. **Long-context discipline.** Tool-calling habits must hold at 50k+
   input tokens, not just on short prompts. The chardet failure fired
   at 71k input tokens, not at the start of the conversation.
4. **Recovery from tool errors.** When a tool returns an error (failed
   path, syntax issue, permission denied), the next response must be
   another tool call, not a confused near-empty response. This is
   what specifically ended the chardet session.
5. **Clean separation in the response shape that LiteLLM's Anthropic
   adapter can map.** `message.content` is the visible text block,
   `message.tool_calls[]` are `tool_use` blocks, `reasoning_content` is
   the `thinking` block. No model-specific markup leaking across
   channels.

Items 1–4 are *training* properties. Item 5 is a *serving-stack*
property (vLLM's tool parser + LiteLLM's translator together). The
existing stack handles item 5 faithfully for any model whose vLLM
parser exists; the gap is items 1–4.

## How the candidates score against that test

### Tier A — explicitly trained for the Claude Code loop

**GLM-4.6 / 4.7 / GLM-5** (Z.ai, open weights, MIT/Apache).
Best-documented direct Claude-Code training. From Z.ai's own docs:
trained on "7T specialized tokens for reasoning, code and agentic
tasks, with curriculum specifically tailored to real-world
requirements through RL — including function calling, web browsing
and tool usage." Z.ai published a 74-test Claude-Code benchmark
where GLM-4.6 surpassed Sonnet 4. GLM-4.7 ships an explicit
"Coding Agent Quickstart" for vLLM. Thinking mode is integrated
with tool-calling — the model is trained to chain reasoning steps
*with* tool calls instead of treating reasoning as terminal.

- vLLM tool parser: `glm45` / `glm4_moe`
- Anthropic-shape surface: documented drop-in for Claude Code env
  vars (`ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN`)
- Risk: low; this is the model whose training pipeline most directly
  targets the failure mode this stack hits

**Kimi K2 / K2.6** (Moonshot, MIT). Trained explicitly for
"agent swarm" and "long-horizon execution" — the most relevant
training story for the chardet failure, which was specifically a
long-horizon recovery break. Moonshot ships their own
Anthropic-compatible API surface; the community-maintained
[`LLM-Red-Team/kimi-cc`](https://github.com/LLM-Red-Team/kimi-cc)
proves the shape works for Claude Code in practice. Public
technical detail on the training is less specific than GLM's, but
the productised Claude-Code fit is real.

- vLLM tool parser: `kimi_k2`
- Risk: low–medium; long-horizon claim is the closest match to the
  empirical failure mode, less independent verification than GLM

**DeepSeek V3.2** (open weights, MIT). The only one on this list
that *ships an Anthropic endpoint as a first-class API* —
`api.deepseek.com/anthropic`. That's evidence Anthropic-shape
responses are a maintained product surface, not an afterthought.
DeepSeek explicitly call out that the V3.2-Speciale variant is
OpenAI-only and *not* Claude-Code-compatible — the same kind of
clean call-out Z.ai makes, which is evidence they know which of
their models pass the bar.

- vLLM tool parser: `deepseek_v3`
- Risk: medium; the reasoning track is separate from the chat track
  and how the two interact mid-task with `tool_use` blocks is less
  documented than GLM's case. Worth disabling reasoning for
  tool-call turns via `extra_body`

### Tier B — agentically trained, family-aware tooling, but Claude-Code positioning is incidental

**Qwen3-Coder (480B-A35B, 30B-A3B, Coder-Next 80B-A3B)**. Trained
explicitly for agentic tool use — Qwen's own docs describe
"executable task synthesis, environment interaction, and RL" for
the Coder line. Strong on the `qwen3_coder` parser, well-tested in
CLINE / Qwen Code / community Claude Code setups. The reason this
is Tier B and not Tier A: the OpenHands comment at `agent.py:651`
flags Qwen specifically for "sometimes places tool-call XML inside
`reasoning_content`" — a family-level quirk that may or may not be
trained out in the Coder variants. The family has the problem, not
just base Qwen3.5.

- vLLM tool parser: `qwen3_coder` (same as base Qwen3) or `qwen3xml`
- Risk: medium; lowest *stack* disruption (drop-in same parser, same
  LiteLLM line) but highest risk of inheriting the family quirk that
  breaks `acp-claude`. Most useful as a baseline to isolate "is it
  the family or the training?"

**gpt-oss-120b / 20b** (OpenAI, Apache-2.0). Trained on tool use as
a first-class output channel via Harmony's 3-channel format
(`analysis` / `commentary` / `final`). The structural separation
makes it harder for the model to "forget" to emit a tool call —
tool calls live in a distinct channel, not in the same token stream
as prose. vLLM has a dedicated server-side agentic loop for
gpt-oss on `/v1/responses` — the *only* family vLLM does that for,
which is independent evidence OpenAI's tool-call format is the
cleanest of the open models.

- vLLM tool parser: `openai` (Harmony)
- Risk: medium; the Harmony channels don't map 1:1 to Anthropic
  content blocks. LiteLLM's `_translate_openai_content_to_anthropic`
  only knows about `content` / `tool_calls` / `reasoning_content`,
  so the mapping flows through vLLM's `openai` tool parser flattening
  Harmony into OpenAI Chat Completions first. Should work, but worth
  a probe before committing — one more translation layer than Tier A

**MiniMax-M2 / M2.1**. Has a vLLM parser (`minimax_m2`); Ollama
lists it as a recommended cloud model. Less Claude-Code-specific
marketing than Tier A, less agentic-RL public detail than
Qwen3-Coder. Worth considering only if Tier A and the rest of
Tier B don't pan out.

### Tier C — general-purpose, likely to suffer the same failure mode as base Qwen3.5

**Llama 3.x / Llama 4, Mistral Large, Gemma 3.** Tool-calling works
mechanically (vLLM has parsers for all of them) but mid-task
tool-call discipline is not a documented training target. Expect
the same "planning sentence → end of turn" pattern as base Qwen3.5.

**Base Qwen3 (non-Coder), DeepSeek V3 base (non-V3.2), GLM-4.5
base.** Same pattern — these are the "instruct" baselines that the
Coder/V3.2/4.6 variants were built *to fix*. The current Qwen3.5
deployment is one of these.

## Recommendation, compatibility-first

Picking based purely on whether `acp-claude` will stay engaged
through a 100-step coding task:

1. **GLM-4.6 (or GLM-4.7 if the vLLM build supports its parser).**
   Strongest evidence of being explicitly trained for the Claude
   Code loop. Z.ai's "74 real-world Claude Code tests" is the most
   concrete vendor claim of "we made this work in your harness
   specifically." Lowest risk that the model will end a turn with a
   planning sentence.
2. **Kimi K2.6.** Long-horizon training is the closest analog to
   what the chardet failure exposed (the model lost the plot after a
   tool error at high context). Anthropic-shape API surface is a
   first-class product.
3. **DeepSeek V3.2 (non-Speciale).** Ships an Anthropic endpoint, so
   the response-shape compatibility is verified by the vendor.
   Reasoning-track interaction is the open question; pair with
   explicit `extra_body` settings to disable reasoning for tool-call
   turns.
4. **Qwen3-Coder-480B-A35B-Instruct or Coder-Next-80B-A3B.** Same
   family as the current deployment, but with the agentic training
   it lacks. Lowest stack disruption (same parser, same LiteLLM
   line). Useful primarily as a *control* — confirms whether the
   `acp-claude` failure is the family or the base-instruct training.
5. **gpt-oss-120b.** Different family, different output format.
   Most likely to work cleanly without prompt-level intervention,
   but the Harmony→Anthropic content-block translation through
   LiteLLM is the largest unknown. Worth a single probe call before
   committing to a full eval.

If only one swap is worth running: **GLM-4.6**, on the strength of
the explicit Claude-Code training pipeline and the most concrete
vendor claim. If that doesn't pan out empirically, **Kimi K2.6**
for the long-horizon training story, then **gpt-oss-120b** as the
"different family entirely" sanity check.

## Implications for the existing `--agent-type` recommendation

The companion doc concludes that `--agent-type default` is the right
answer for qwen3.5. The compatibility lens reframes that:

- For **Tier C backends** (base Qwen, Llama, Mistral, Gemma): keep
  `--agent-type default`. The default agent's `agent.py:651` nudge
  is load-bearing.
- For **Tier B backends** (Qwen3-Coder, gpt-oss): probably still
  `default`, pending empirical check. The structural improvements
  reduce but may not eliminate the failure modes the nudge papers
  over.
- For **Tier A backends** (GLM-4.6 / Kimi K2.6 / DeepSeek V3.2 with
  reasoning disabled): `--agent-type acp-claude` should actually be
  competitive — and that's the experiment most worth running,
  because if it works, the full Claude Code tool ecosystem
  (subagents, MCP, the project's settings.json permissions model)
  becomes available on top of a self-hosted model.

## Sources

- [Z.ai GLM-4.6 blog (74 Claude-Code tests, agentic RL curriculum)](https://z.ai/blog/glm-4.6)
- [Z.ai GLM-4.6 developer docs](https://docs.z.ai/guides/llm/glm-4.6)
- [GLM-4.6 tool-calling technical analysis (cirra.ai)](https://cirra.ai/articles/glm-4-6-tool-calling-mcp-analysis)
- [zai-org/GLM-4.5 (Agentic, Reasoning, Coding foundation models)](https://github.com/zai-org/GLM-4.5)
- [GLM-4.7 vLLM coding-agent quickstart](https://stable-learn.com/en/glm-47-tech-guide/)
- [Kimi K2.6 announcement](https://www.kimi.com/blog/kimi-k2-6)
- [LLM-Red-Team/kimi-cc (Kimi K2 in Claude Code)](https://github.com/LLM-Red-Team/kimi-cc)
- [DeepSeek V3.2 Claude Code integration (Anthropic endpoint)](https://api-docs.deepseek.com/quick_start/agent_integrations/claude_code)
- [DeepSeek V3.2 in Claude Code (Speciale not supported)](https://apidog.com/blog/deepseek-v3-2-deepseek-v3-2-speciale-in-claude-code/)
- [QwenLM/Qwen3-Coder repo (agentic training, vLLM parser)](https://github.com/QwenLM/Qwen3-Coder)
- [Qwen3-Coder-30B-A3B-Instruct model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
- [vLLM GPT-OSS recipe (Harmony, openai tool parser)](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html)
- [vLLM tool-call-parser list](https://docs.vllm.ai/en/latest/features/tool_calling/)
- [musistudio/claude-code-router (community middleware with `enhancetool` / `reasoning` transformers)](https://github.com/musistudio/claude-code-router)
