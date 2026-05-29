# Open-source LLMs and the Claude Code agent loop: a compatibility evaluation

> **TL;DR.** Claude-Code compatibility hinges on one model behaviour:
> emitting tool-call markup (which becomes `stop_reason: tool_use`) on
> every turn where the task isn't done. We have one observed failure
> mode — qwen3.5 in the chardet trace — where the model hits EOS
> mid-task without that markup. Vendors that advertise agentic-RL
> training and ship hosted Anthropic-format endpoints (GLM-4.6/4.7,
> Kimi K2.6, DeepSeek V3.2) are the most likely candidates to avoid
> the same failure self-hosted, but we have no direct test data on
> any of them through `acp-claude` yet. Models without agentic-RL
> training claims (base Qwen3, and by analogy other general-purpose
> open models) may exhibit the same pattern, pending empirical check.

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

Claude Code, like any Anthropic-API client, terminates a turn when
the response carries `stop_reason: "end_turn"` — the spec's
authoritative signal that the model is done. There is no
client-side heuristic to override; the same rule that lets Claude
itself finish a conversation cleanly is what terminates a session
early on a misbehaving backend. The only lever the model has to keep
the loop alive is to emit tool-call markup so the upstream
`finish_reason` becomes `tool_calls`, which LiteLLM maps to
`stop_reason: tool_use`. (OpenAI Chat Completions has no analogue of
Anthropic's `pause_turn`, so the bridge has no third-way escape.)

The chardet failure pinned down five behaviours a backend must get
right for `acp-claude` to not terminate sessions early:

1. **Mid-task tool-call discipline.** Every "let me now look at X"
   sentence must be emitted *with* a `<tool_call>` markup token in
   the same generation, so the upstream `finish_reason` is
   `tool_calls` (→ Anthropic `stop_reason: tool_use`) rather than
   `stop` (→ `end_turn`). Hitting EOS after text alone, mid-task, is
   the single dominant failure mode.
2. **Reasoning that doesn't swallow tool calls.** If the model uses a
   `<think>`-style channel, the tool-call markup still has to appear
   in the visible output channel — not get embedded in the reasoning
   trace where the parser strips it (which leaves no markup → `stop`
   → `end_turn`). This is the specific Qwen-class regression the
   OpenHands default agent nudges around at `agent.py:651`.
3. **Long-context discipline.** The discipline from #1 must hold at
   50k+ input tokens, not just on short prompts. The chardet failure
   fired at 71k input tokens, not at the start of the conversation.
4. **Recovery from tool errors.** When a tool returns an error
   (failed path, syntax issue, permission denied), the next response
   must again include tool-call markup, not a confused near-empty
   text response. This is what specifically ended the chardet session.
5. **Clean separation in the response shape that LiteLLM's Anthropic
   adapter can map.** `message.content` is the visible text block,
   `message.tool_calls[]` are `tool_use` blocks, `reasoning_content`
   is the `thinking` block. No model-specific markup leaking across
   channels.

Items 1–4 are *training* properties (the model has to learn when to
emit the markup). Item 5 is a *serving-stack* property (vLLM's tool
parser + LiteLLM's translator together). The existing stack handles
item 5 faithfully for any model whose vLLM parser exists; the gap is
items 1–4, all of which collapse to "the model must signal
`tool_use`, not `end_turn`, whenever the task is unfinished."

## How the candidates score against that test

### Tier A — vendors that publicly position the model for Claude Code

These are open-weight models whose vendors (a) advertise agentic-RL
training and (b) operate a hosted Anthropic-format endpoint they
market for Claude Code use. The hosted endpoint is *not* the
self-hosting path (that goes via vLLM + LiteLLM as usual), but its
existence is evidence the vendor has built and maintains a product
around Claude-Code-shape responses.

**GLM-4.6 / 4.7** (Z.ai, MIT-licensed weights). Z.ai's docs describe
training on "7T specialized tokens for reasoning, code and agentic
tasks, with curriculum specifically tailored to real-world
requirements through RL — including function calling, web browsing
and tool usage." Z.ai also publishes a vendor-self-reported "74-test
Claude-Code benchmark" where GLM-4.6 surpasses Sonnet 4; we have not
independently reproduced it. GLM-4.7 ships an explicit "Coding
Agent Quickstart" for vLLM.

- vLLM tool parser: `glm45` / `glm4_moe`
- Hosted Anthropic endpoint: `https://api.z.ai/api/anthropic`
- Notes: training pipeline is the most directly described for this
  use case among the open models we checked; no independent
  acp-claude test data

**Kimi K2 / K2.6** (Moonshot, MIT-licensed weights). Vendor markets
"agent swarm" and "long-horizon execution" as training targets but
does not publish a Claude-Code-specific benchmark. The community
[`LLM-Red-Team/kimi-cc`](https://github.com/LLM-Red-Team/kimi-cc)
project sets `ANTHROPIC_BASE_URL` to Moonshot's hosted endpoint;
that confirms Moonshot's *hosted* endpoint serves Claude Code
correctly, not the self-hosted vLLM path.

- vLLM tool parser: `kimi_k2`
- Hosted Anthropic endpoint: `https://api.moonshot.ai/anthropic`
  (also applies a static `real_temperature = request_temperature *
  0.6` rescale)
- Known issue: vLLM issue #30238 reports self-hosted tool-call
  failures for the **Kimi-K2-Thinking** variant specifically;
  non-thinking K2-Instruct is the safer self-hosting target

**DeepSeek V3.2** (MIT-licensed weights). The only Tier-A vendor
that documents the hosted-endpoint translation in detail:
[explicit content-block support matrix](https://api-docs.deepseek.com/guides/anthropic_api)
(text + tool_use supported; image / document / search_result /
server tool / web_search / MCP tool / container upload NOT
supported), and Claude model-name aliasing
(`claude-opus*` → `deepseek-v4-pro`, `claude-sonnet*` /
`claude-haiku*` → `deepseek-v4-flash`). DeepSeek separately
documents that the V3.2-Speciale variant is OpenAI-only and not
served through their Anthropic endpoint.

- vLLM tool parser: `deepseek_v3`
- Hosted Anthropic endpoint: `https://api.deepseek.com/anthropic`
- Notes: reasoning track is structurally separate from the chat
  track; impact on mid-task `tool_use` discipline is not
  documented in vendor sources and would need empirical check

### Tier B — agentically trained, but no Claude-Code-specific vendor positioning

**Qwen3-Coder (480B-A35B, 30B-A3B, Coder-Next 80B-A3B)**. Qwen's
docs describe "executable task synthesis, environment interaction,
and RL" for the Coder line. Strong tool-use story in CLINE / Qwen
Code / community Claude Code setups (we did not test these
ourselves). The reason this is Tier B and not Tier A: the OpenHands
comment at `agent.py:651` flags Qwen specifically for "sometimes
places tool-call XML inside `reasoning_content`" — a documented
family-level quirk for *some* Qwen3 model. Whether Qwen3-Coder
shares it is unknown without direct testing.

- vLLM tool parser: `qwen3_coder` (same as base Qwen3) or `qwen3xml`
- Notes: lowest stack disruption (same parsers and LiteLLM model
  line as the current Qwen3.5 deployment). Useful primarily as a
  *control* — confirms whether the `acp-claude` failure is the
  family or the base-instruct training.

**gpt-oss-120b / 20b** (OpenAI, Apache-2.0). Trained on tool use as
a first-class output channel via Harmony's 3-channel format
(`analysis` / `commentary` / `final`). Whether the channel
separation translates to better mid-task `tool_use` discipline in
practice is unverified. vLLM gives gpt-oss a dedicated server-side
agentic loop on `/v1/responses` — the only family vLLM does this
for, which we read as evidence that OpenAI's tool-call format has
distinct serving requirements, not as a quality judgement on the
model.

- vLLM tool parser: `openai` (Harmony)
- Notes: Harmony channels don't map 1:1 to Anthropic content
  blocks. LiteLLM's `_translate_openai_content_to_anthropic` only
  knows about `content` / `tool_calls` / `reasoning_content`, so
  the mapping flows through vLLM's `openai` tool parser flattening
  Harmony into OpenAI Chat Completions first. Worth a probe before
  committing.

**MiniMax-M2 / M2.1**. Has a vLLM parser (`minimax_m2`); Ollama
lists it as a recommended cloud model. We did not find
Claude-Code-specific marketing or independent benchmarks; included
for completeness.

### Tier C — general-purpose instruct models without published agentic-RL training

**Llama 3.x / Llama 4, Mistral Large, Gemma 3.** Tool-calling works
mechanically (vLLM has parsers for all of them); we did not find
vendor documentation of mid-task tool-call discipline as a training
target. By analogy to qwen3.5 they *may* exhibit the same
"planning sentence → end of turn" pattern, but we have not tested
them through `acp-claude`.

**Base Qwen3 (non-Coder), DeepSeek V3 base (non-V3.2), GLM-4.5
base.** Same category — instruct baselines for the Coder/V3.2/4.6
variants. The current Qwen3.5 deployment is one of these and is the
only one we have direct trace data for. Whether the others share
the failure mode is a reasonable guess, not an observation.

## Suggested experiment order

Ranked by strength of *publicly available evidence* that the model
might survive `acp-claude` — not by tested outcome, since we have
not tested any of these.

1. **GLM-4.6 (or GLM-4.7 if the vLLM build supports its parser).**
   The only candidate with a vendor-published Claude-Code-specific
   benchmark (74 tests, self-reported); training-recipe description
   names function calling and tool usage explicitly.
2. **Kimi K2-Instruct (avoid K2-Thinking on self-host per vLLM #30238).**
   Vendor markets long-horizon agentic training; the hosted endpoint
   is widely used with Claude Code via `kimi-cc`, which is evidence
   for the hosted endpoint but not for self-host parity.
3. **DeepSeek V3.2 (non-Speciale).** Vendor documents the
   Anthropic-shape translation in detail (content-block matrix,
   model aliases). Reasoning-track interaction with `tool_use` is
   undocumented.
4. **Qwen3-Coder-480B-A35B-Instruct or Coder-Next-80B-A3B.**
   Smallest stack change. Tests whether agentic training within the
   same family fixes the failure we observed in qwen3.5 — i.e.
   isolates "family" from "training recipe."
5. **gpt-oss-120b.** Different family, different output channel
   structure. Worth a probe call to confirm the Harmony →
   Anthropic content-block translation through LiteLLM works
   cleanly before a full eval.

The first three are the most worth testing because each has at
least one vendor-published artefact pointing at Claude-Code
compatibility. The last two are useful as controls — same-family
and different-family alternatives to the current deployment.

## Implications for the existing `--agent-type` recommendation

The companion doc concludes that `--agent-type default` is the right
answer for qwen3.5. The compatibility lens reframes that as a
per-tier set of hypotheses to test, not as new recommendations:

- For **Tier C backends** (base Qwen, Llama, Mistral, Gemma): keep
  `--agent-type default` for now. The default agent's `agent.py:651`
  nudge demonstrably rescues qwen3.5; analogous behaviour is
  plausible but untested for the others.
- For **Tier B backends** (Qwen3-Coder, gpt-oss): no recommendation
  without empirical data. Both have arguments for being more robust
  than qwen3.5, but neither has been tested through `acp-claude` in
  this stack.
- For **Tier A backends** (GLM-4.6, Kimi K2-Instruct, DeepSeek V3.2):
  worth testing `--agent-type acp-claude` directly. If it works,
  the full Claude Code tool ecosystem (subagents, MCP, the
  project's `settings.json` permissions model) becomes available on
  top of a self-hosted model. If it doesn't, fall back to
  `--agent-type default`.

## Sources

Primary (cited directly):

- [Z.ai GLM-4.6 blog — vendor-self-reported 74-test Claude-Code benchmark, agentic-RL curriculum description](https://z.ai/blog/glm-4.6)
- [Z.ai GLM-4.6 developer docs](https://docs.z.ai/guides/llm/glm-4.6)
- [GLM-4.7 vLLM coding-agent quickstart](https://stable-learn.com/en/glm-47-tech-guide/)
- [Kimi K2.6 announcement (Moonshot)](https://www.kimi.com/blog/kimi-k2-6)
- [LLM-Red-Team/kimi-cc — sets ANTHROPIC_BASE_URL to Moonshot hosted endpoint](https://github.com/LLM-Red-Team/kimi-cc)
- [vLLM Kimi-K2.5 recipe — documents hosted-vs-self-host temperature scaling and thinking-mode flags](https://docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-K2.5.html)
- [vLLM issue #30238 — Kimi-K2-Thinking self-host tool-call failure](https://github.com/vllm-project/vllm/issues/30238)
- [DeepSeek Anthropic API docs — content-block support matrix and Claude-model aliasing](https://api-docs.deepseek.com/guides/anthropic_api)
- [DeepSeek V3.2 in Claude Code (Speciale not supported)](https://apidog.com/blog/deepseek-v3-2-deepseek-v3-2-speciale-in-claude-code/)
- [QwenLM/Qwen3-Coder repo — agentic training description](https://github.com/QwenLM/Qwen3-Coder)
- [Qwen3-Coder-30B-A3B-Instruct model card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
- [vLLM tool-call-parser list (all `--tool-call-parser` names cited)](https://docs.vllm.ai/en/latest/features/tool_calling/)
- [vLLM GPT-OSS recipe (Harmony, `openai` tool parser)](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html)

Background (not directly cited but relevant):

- [musistudio/claude-code-router — community middleware with `enhancetool` / `reasoning` transformers](https://github.com/musistudio/claude-code-router)
- [zai-org/GLM-4.5 repo](https://github.com/zai-org/GLM-4.5)
