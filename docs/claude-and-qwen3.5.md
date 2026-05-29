# Claude Code agent loop vs. qwen3.5: a turn-termination mismatch

> **TL;DR.** Claude Code ends an agent turn when the response carries
> `stop_reason: end_turn` — the Anthropic API's authoritative end-of-turn
> signal. qwen3.5 routinely sends that signal mid-task: it generates a
> planning sentence (or an empty post-error response) and hits EOS
> without emitting any `<tool_call>` markup, so vLLM correctly sets
> `finish_reason: stop`, LiteLLM correctly maps it to
> `stop_reason: end_turn`, and Claude Code correctly believes the model
> is done. The bridge is faithful and Claude Code is honoring the
> contract — the model is the one signalling end-of-turn when it shouldn't.

When `--agent-type acp-claude` is paired with a non-Claude backend model
(here, `qwen3.5` routed through a LiteLLM proxy), the Claude Code agent
loop terminates a session well before the task is done. The result is that
the agent rarely edits any code, even on instances the same model can
handle when driven by `--agent-type default`.

This document records what was observed across one full 16-instance run of
the commit0 lite split (`wentingzhao/commit0_combined`) on the
`acp-claude` agent with qwen3.5 as the backend.

## What we saw

Across all 16 instances of the lite split:

- **15 of 16 instances had zero `edit` tool calls** — the agent never
  modified a file. The only exception was `deprecated`, which produced 7
  edits before terminating mid-thought.
- **Every instance ended with exactly one `FinishAction`** regardless of
  whether the task was complete.
- **The finish messages read as preambles, not conclusions.** They are
  the kind of "let me now do X" sentence a model writes before its next
  batch of tool calls — and then the next batch never happens.

Sample finish messages (verbatim, truncated):

| Instance     | FinishAction message                                                                                        |
|--------------|--------------------------------------------------------------------------------------------------------------|
| chardet      | "Now I understand the codebase better. Let me look at the remaining prober files ... and then implement..." |
| pyjwt        | "I'll start by exploring the repository structure ... Let me read all the main JWT library so..."            |
| jinja        | "There's a syntax error in the code. Let me look at that file:"                                              |
| portalocker  | "I'll help you complete the implementations for the portalocker package. Let me s..."                       |
| deprecated   | "Now I have a clear picture of what needs to be implemented. Let me create a task to track progress: ..."   |

In every case, the model declared intent to act, then yielded back to the
caller without acting.

## Why this happens

Claude Code's agent loop honors the Anthropic API contract for turn
termination: when the response carries `stop_reason: "end_turn"`, the
loop treats the turn as complete and returns from `prompt()`. The
Anthropic spec defines exactly four stop reasons a client can
distinguish: `end_turn` (model is done), `tool_use` (model wants to
invoke a tool — keep the loop going), `max_tokens` (cap hit), and
`stop_sequence`. There is no upstream OpenAI Chat Completions
equivalent of a "I'm not done but please come back to me" signal —
the closest, `pause_turn`, only exists on the Anthropic Messages
endpoint and a translator like LiteLLM has no upstream value to map
into it.

The signalling chain for our stack is fully deterministic:

1. The model decides whether to emit `<tool_call>...</tool_call>`
   markup before EOS.
2. vLLM's `qwen3_coder` tool parser sets `finish_reason: "tool_calls"`
   iff that markup was present; otherwise `"stop"` (or `"length"` if
   the token cap was hit).
3. LiteLLM's adapter maps `tool_calls → tool_use` and `stop → end_turn`
   (`adapters/transformation.py:_translate_openai_finish_reason_to_anthropic`).
4. Claude Code's loop terminates iff it receives `stop_reason: end_turn`.

So the model has exactly one lever for keeping the loop alive: emit
the `<tool_call>` markup. Against Claude this is implicit — the model
is trained to keep `tool_use` blocks paired with any mid-task text,
which produces `tool_use` stop_reason and keeps the loop going.
qwen3.5 is not trained the same way. It will end a generation with a
text-only "let me now look at..." message and hit EOS without the
markup, at which point every layer of the stack is doing the right
thing and the conversation still ends.

Mechanically:

1. Agent: a few `Glob` / `Read` / `Bash` tool calls (exploration).
2. Agent: text message "Now let me implement X..." with no `tool_use`.
3. Claude Code: "model finished its turn" → `prompt()` returns.
4. `ACPAgent.step()` returns.
5. `ACPAgent` emits a terminal `FinishAction` to delimit the completed
   remote turn (this is how `ACPAgent` signals end-of-step to OpenHands).
6. The OpenHands conversation loop sees `FinishAction` and terminates.

`max_iterations=100` does not help. With `acp-claude`, each OpenHands
iteration is one full Claude Code session, and qwen3.5 ends the session
in 2-30 tool calls.

## Wire-level evidence (and what triggers it in practice)

The LiteLLM Anthropic-format adapter
(`litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py`,
`_translate_openai_content_to_anthropic` and
`translate_openai_response_to_anthropic`) faithfully maps the upstream
OpenAI-shape response back to Anthropic content blocks: `reasoning_content
→ thinking`, `message.content → text`, each `tool_calls[i] → tool_use`,
and `finish_reason: tool_calls → stop_reason: tool_use`. It does not
drop, re-order, or coalesce blocks, and it cannot synthesise a `tool_use`
the model did not emit. Live probes of the deployed
`POST /v1/messages` route with `qwen3.5` confirm: when the model returns
text + tool_call together, the Anthropic response carries both.

So the mismatch is not in the bridge. It is between the model's training
and the Anthropic API's `stop_reason` contract that Claude Code honors:

- **Claude is trained** to keep `tool_use` blocks in the same response
  whenever it intends to continue (which produces
  `stop_reason: tool_use`), and to send `stop_reason: end_turn` only
  when the task is genuinely done.
- **qwen3.5 is not.** It will end a generation with a text-only message
  ("Now I understand the codebase better. Let me look at..."), a
  reasoning-only message (everything consumed by the `qwen3` reasoning
  parser into `reasoning_content`, visible `content` empty), or a short
  near-empty response after a failed tool call.

Each of those produces `finish_reason: stop` upstream, which is faithfully
mapped to `stop_reason: end_turn` for the Anthropic-format response —
and that is the authoritative signal Claude Code obeys.

In the chardet trace the chain of events at the wire was:

1. LLM call at `08:07:20.237` (Laminar span; 32 k input tokens) returns
   text **and** `tool_use: Read /workspace/chardet/chardet/multicharsetprober.py`.
   That filename is hallucinated — the real prober files are
   `mbcsgroupprober.py`, `sbcharsetprober.py`, etc.
2. Tool returns `"File does not exist. Note: your current working
   directory is /workspace/chardet."`.
3. LLM call at `08:07:24.039` (71 k input tokens, **9 output tokens**,
   `is_streaming: true`, no `gen_ai.completion.0.function_call.*`
   attributes; upstream `finish_reason: stop` → Anthropic
   `stop_reason: end_turn`) returns a response with no `tool_use` block
   and an explicit end-of-turn signal.
4. Claude Code's `prompt()` returns (per the Anthropic contract on
   `end_turn`); `ACPAgent.step()` emits `FinishAction`; OpenHands exits
   at `08:07:27.385`.

(Note: the Laminar OTel span shows `finish_reasons: ["completed"]`.
That is the proxy-layer normalisation of upstream OpenAI `"stop"`, not
a separate state — the raw model response had no `<tool_call>` markup,
vLLM cleanly set `finish_reason: stop`, and LiteLLM mapped to
`stop_reason: end_turn`.)

So the "preamble text, then nothing" pattern documented above has a
sharper variant worth flagging: **tool-failure recovery at high context**
also triggers it. qwen3.5 recovers cleanly from a tool error in a small
context (verified by a direct probe with the same hallucinated path —
the model emitted `"Let me try a different path."` and a corrected
`Read`), but at 71 k tokens it produced 9 tokens of nothing usable and
the session ended.

## How `--agent-type default` survives the same model output

`acp-claude` cedes turn control entirely to the Claude Code subprocess
— which uses the Anthropic `stop_reason` as the authoritative end-of-turn
signal, as any Anthropic-API client should. The default agent does not
defer that decision to the model's `stop_reason`. It owns the loop in
`openhands-sdk/openhands/sdk/agent/agent.py:Agent.step()` and inspects
the message *shape* directly (text content, tool calls, reasoning
blocks), ignoring `stop_reason` and applying its own policy:

| Model response                                         | Default agent action                                                                                                                            | `acp-claude` action |
|--------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| `tool_calls` present (stop_reason: tool_use)           | Execute them, loop.                                                                                                                              | Claude Code executes them, loop within the SDK. |
| `has_content=True`, no `tool_calls` (stop_reason: end_turn) | Emit `MessageEvent`, set `execution_status = FINISHED` (treats as awaiting user input).                                                          | Claude Code returns on `end_turn`; `FinishAction` emitted. |
| `has_content=False`, no `tool_calls` (stop_reason: end_turn, reasoning-only or empty) | Emit a synthetic user-side nudge — *"Your last response did not include a function call or a message. Please use a tool to proceed with the task."* — and **continue the loop** despite the `end_turn` signal. | Claude Code returns on `end_turn`; `FinishAction` emitted. |

The third row is the load-bearing difference. The default agent has
explicit, Qwen-aware corrective feedback for the empty/reasoning-only
case (`agent.py:651` — the source comment names Qwen specifically:
*"common with Qwen, which sometimes places tool-call XML inside
reasoning_content"*). That matches the dominant qwen3.5 failure mode
observed in the wire trace (9-token, no-tool-call response after a tool
error, `stop_reason: end_turn`), and converts what would be a
session-ender into a re-prompt — at the cost of overriding the model's
own end-of-turn declaration, which is the kind of contract violation
`acp-claude` cannot do without breaking the spec for Claude itself.

The text-only-planning case (row 2) is not actually rescued by the
default agent — it also finishes there. The reason the default agent
still scored higher in head-to-head runs is structural:

- **Iteration budget.** With `--agent-type default`, one OpenHands
  iteration = one LLM call. `max_iterations=100` buys 100 LLM calls and
  the loop persists across reasoning-only responses via the nudge above.
- **With `acp-claude`**, one OpenHands iteration = one full Claude Code
  session. The SDK ends each session on the first text-only / empty
  response, so 100 iterations buys at most 100 short sessions — each of
  which has to re-acquire its own working state because the ACP session
  is fresh.

The default agent's headline failure mode (*"agent reached maximum
iterations limit (100)"*) is the consequence of this: qwen3.5 stays
*engaged* through many LLM turns and eventually exhausts the budget
mid-work, instead of being declared done after a planning sentence.

## Why the report.json scores are misleading

The aggregate report.json showed 8/16 "completed" instances scoring some
tests passing. None of those were the agent's work:

| Instance     | Edits | Tests passed   | Why a score?                                          |
|--------------|------:|----------------|--------------------------------------------------------|
| cachetools   |     0 | 153/215 (71%) | Repo imports cleanly; many tests do not exercise stubs |
| simpy        |     0 |  82/140 (59%) | Same pattern                                           |
| chardet      |     0 |   7/381 (2%)  | Same idea, but tests do exercise the unfilled code     |
| pyjwt        |     0 |   1/1 (100%)  | Degenerate: pytest collected only 1 test               |
| wcwidth      |     0 |   0/39        | Tests fail because stubs are unfilled                  |
| voluptuous   |     0 |   0/1         | Same                                                   |
| babel        |     0 |   0/4         | Same                                                   |
| deprecated   |     7 |  35/171 (20%) | Real engagement, but cut short mid-edit-cycle          |

The "completions" are baseline pass rates of the unmodified repo. The
`pyjwt` "100%" is a hollow win — pytest only collected one test against
a repo that the agent never touched.

The 8 "errored" instances split into two failure modes, both reflecting
the unmodified base repo rather than agent damage:

- **"Report summary extraction failed (exit code 1)"** — pytest crashed
  before producing `report.json` (typically a collection-time
  `SyntaxError` or `TypeError` from importing unfilled stubs). Instances:
  `tinydb`, `marshmallow`, `jinja`, `cookiecutter`, `portalocker`.
- **"Report summary missing or empty 'total' field"** — pytest produced
  a report but collected zero tests for the same underlying reason.
  Instances: `imapclient`, `parsel`, `minitorch`.

## The one engaged case: `deprecated`

`deprecated` is the most useful counterexample. The agent issued 31 tool
calls (7 edits, 13 executes, 10 reads), edited `classic.py` repeatedly,
ran targeted pytest commands, and even wrote `python -c` probes to
inspect Python's introspection behavior. Despite all of that, the
conversation still terminated mid-thought:

> "Now I have a clear picture of what needs to be implemented. Let me
> create a task to track progress: I need to fix the decorator logic -
> it should det..."

The default agent (`--agent-type default`) on the same model achieved
171/171 (100%) on `deprecated`. The `acp-claude` agent achieved 35/171
(20%). So even on the one instance where qwen3.5 stayed engaged inside
Claude Code, the loop terminated before convergence.

## Comparison to `--agent-type default`

Same model, same prompt, same `max_iterations=100`, same retry budget,
different driver:

| Configuration                      | Resolved | Test-pass rate (head-to-head instances)            |
|------------------------------------|---------:|----------------------------------------------------|
| `--agent-type default` (run A)     |     1/16 | 46.6% (527/1131)                                   |
| `--agent-type acp-claude` (run B)  |     1/16 | 29.2% (278/952)                                    |

On the four instances both runs legitimately attempted (cachetools,
chardet, deprecated, wcwidth), the default agent wins every one.
`deprecated` is the most dramatic: A solved it cleanly, B got 20.5% and
unresolved.

A's headline failures are "agent reached maximum iterations limit (100)"
— the model running too long without converging. B's headline failures
are "agent never wrote code" — the loop terminating after a few tool
calls plus a planning sentence.

## Implications

- **`acp-claude` honors the Anthropic API contract; the failure is the
  model.** Claude Code terminates on `stop_reason: end_turn` because
  that is exactly what the spec tells it to do — and it has to, because
  Claude itself relies on the same signal to delimit its turns. Pairing
  it with a non-Claude model that sends `end_turn` mid-task (text-only
  "planning" responses, reasoning-only responses, near-empty
  post-tool-error responses) will cause premature session termination on
  most instances. Increasing `--max-iterations` does not help; it does
  not affect the in-session loop that Claude Code runs.
- **LiteLLM is not the gap.** The Anthropic-format adapter faithfully
  translates the upstream OpenAI shape to Anthropic content blocks and
  finish-reason values; the reason `stop_reason: end_turn` reaches
  Claude Code is that the model emitted no `<tool_call>` markup and hit
  EOS cleanly, which means `finish_reason: stop` upstream and
  `stop_reason: end_turn` downstream is the *correct* translation.
  There is no `pause_turn`-style escape hatch in OpenAI Chat
  Completions for the bridge to compose with.
- **For qwen3.5 (and likely other non-Claude backends), use
  `--agent-type default`.** Its `step()` has explicit handling for
  reasoning-only / empty-content responses (a synthetic user-side nudge
  back to action — see `agent.py:651`), which matches qwen3.5's most
  common failure mode. The default agent also gets a real
  iteration budget because each OpenHands iteration is one LLM call,
  not one whole subprocess session.
- **Fixing this without changing the model would require a change at
  the Claude Code level** (have the session continue past text-only
  turns, or have the model adapter never emit text without a tool
  call). Both are outside this repo's reach — the SDK lives in
  `vendor/software-agent-sdk` and the Claude Code adapter lives in the
  upstream `@agentclientprotocol/claude-agent-acp` package.

## Reproduction notes

- Dataset: `wentingzhao/commit0_combined`, split `test`, repo-split
  `lite` (16 instances).
- Model: `litellm_proxy/qwen3.5` via substrate LiteLLM proxy.
- Eval output dir for the acp-claude run analyzed here:
  `eval_outputs/wentingzhao__commit0_combined-lite/litellm_proxy/qwen3.5_sdk_3e0a3a0_maxiter_100_N_claude-test/`
- Conversation event archives per instance:
  `conversations/<instance>.tar.gz` inside the eval output dir. Extract
  and inspect `workspace/conversations/*/events/event-*.json` for the
  full per-instance event stream that supports every claim above.
