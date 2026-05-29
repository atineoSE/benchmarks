# Claude Code agent loop vs. qwen3.5: a turn-termination mismatch

> **TL;DR.** Claude Code ends an agent turn when the assistant message has
> no `tool_use` block; qwen3.5 routinely produces such messages mid-task
> (planning text, reasoning-only chunks, or short empty responses after a
> failed tool call), so the session ends before the model writes any code.
> The LiteLLM Anthropic-format bridge is faithful — it does not introduce
> the gap; the mismatch is between the model's training and Claude Code's
> turn-termination rule.

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

Claude Code's agent loop uses a simple turn-termination heuristic: when
the assistant message contains text but no `tool_use` blocks, the loop
treats the turn as complete and returns from `prompt()`. That is the
right call against Claude, which is trained to either

- continue acting with `tool_use` blocks in the same response, or
- emit a text-only response only when it has nothing left to do.

qwen3.5 does not follow this convention. It interleaves text-only
"planning" messages mid-task — the very pattern Claude would inline with
the next tool call. Claude Code reads that text-only response as
end-of-turn, returns from the ACP session, and the conversation is over.

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
and Claude Code's turn-termination rule:

- **Claude is trained** to keep `tool_use` blocks in the same response
  whenever it intends to continue, and to emit text-only only when the
  task is genuinely done.
- **qwen3.5 is not.** It will end a generation with a text-only message
  ("Now I understand the codebase better. Let me look at..."), a
  reasoning-only message (everything consumed by the `qwen3` reasoning
  parser into `reasoning_content`, visible `content` empty), or a short
  near-empty response after a failed tool call.

Each of those translates to an Anthropic response with no `tool_use`
block, which Claude Code interprets as end-of-turn.

In the chardet trace the chain of events at the wire was:

1. LLM call at `08:07:20.237` (Laminar span; 32 k input tokens) returns
   text **and** `tool_use: Read /workspace/chardet/chardet/multicharsetprober.py`.
   That filename is hallucinated — the real prober files are
   `mbcsgroupprober.py`, `sbcharsetprober.py`, etc.
2. Tool returns `"File does not exist. Note: your current working
   directory is /workspace/chardet."`.
3. LLM call at `08:07:24.039` (71 k input tokens, **9 output tokens**,
   `is_streaming: true`, no `gen_ai.completion.0.function_call.*`
   attributes) returns a response with no `tool_use` block.
4. Claude Code's `prompt()` returns; `ACPAgent.step()` emits
   `FinishAction`; OpenHands exits at `08:07:27.385`.

So the "preamble text, then nothing" pattern documented above has a
sharper variant worth flagging: **tool-failure recovery at high context**
also triggers it. qwen3.5 recovers cleanly from a tool error in a small
context (verified by a direct probe with the same hallucinated path —
the model emitted `"Let me try a different path."` and a corrected
`Read`), but at 71 k tokens it produced 9 tokens of nothing usable and
the session ended.

## How `--agent-type default` survives the same model output

`acp-claude` cedes turn control entirely to the Claude Code subprocess.
The default agent does not — it owns the loop in
`openhands-sdk/openhands/sdk/agent/agent.py:Agent.step()` and inspects
the message itself. The three cases it distinguishes:

| Model response                                         | Default agent action                                                                                                                            | `acp-claude` action |
|--------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| `tool_calls` present                                   | Execute them, loop.                                                                                                                              | Claude Code executes them, loop within the SDK. |
| `has_content=True`, no `tool_calls`                    | Emit `MessageEvent`, set `execution_status = FINISHED` (treats as awaiting user input).                                                          | Claude Code returns; `FinishAction` emitted. |
| `has_content=False`, no `tool_calls` (reasoning-only or empty) | Emit a synthetic user-side nudge — *"Your last response did not include a function call or a message. Please use a tool to proceed with the task."* — and **continue the loop**. | Claude Code returns; `FinishAction` emitted. |

The third row is the load-bearing difference. The default agent has
explicit, Qwen-aware corrective feedback for the empty/reasoning-only
case (`agent.py:651` — the source comment names Qwen specifically:
*"common with Qwen, which sometimes places tool-call XML inside
reasoning_content"*). That matches the dominant qwen3.5 failure mode
observed in the wire trace (9-token, no-tool-call response after a tool
error), and converts what would be a session-ender into a re-prompt.

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

- **`acp-claude` is built around Claude's response shape.** Pairing it
  with a non-Claude model that emits mid-stream planning text, a
  reasoning-only message, or a near-empty response after a tool error
  will cause premature session termination on most instances.
  Increasing `--max-iterations` does not help; it does not affect the
  in-session loop that Claude Code runs.
- **LiteLLM is not the gap.** The Anthropic-format adapter faithfully
  translates the upstream OpenAI shape to Anthropic content blocks; the
  reason no `tool_use` reaches Claude Code is that the model did not
  emit one.
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
