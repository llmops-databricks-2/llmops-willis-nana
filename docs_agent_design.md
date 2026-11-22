# Agent design: sales-call assistant

## Use case
A sales manager (here: the DataMentoring prospect calls) asks questions about recorded calls, in a conversation.

## Tools
| Tool | Type | Used for |
|---|---|---|
| `query_space_<id>` / `poll_response_<id>` | MCP (Genie space "Sales Calls Analytics" on `call_analysis`) | counts, dates, durations, speakers |
| `search_call_transcripts` | custom function (pgvector on `call_chunks_turns_synced`, Lakebase) | what was said: needs, objections, quotes |

## Orchestration (per turn)
| # | Step | Who decides |
|---|---|---|
| 1 | Load the last 20 messages of the session from Lakebase (`agent_messages`) | code |
| 2 | Call the LLM with system prompt + history + question + tool specs | code |
| 3 | LLM answers, or requests tools: which tool, which arguments, when to stop | **model** |
| 4 | Run each requested tool, feed results back; repeat 2-3, at most 5 steps | code (loop bound, error capture) |
| 5 | Save the user question and final answer to Lakebase | code |

## Design choices
- **Hand-written loop** (OpenAI client through the AI Gateway): every code-controlled step is visible; no framework dependency.
- **Memory = one row per message**, user/assistant only. Tool calls stay inside the turn, so history stays small and always valid for the chat API.
- **Two tools with distinct descriptions** so the model routes analytical vs. content questions; the Genie space hides the transcript column to avoid bloating its context.
- **Tool errors are returned to the model** as text instead of raising, so it can retry or answer honestly.
- **MLflow tracing** on the turn and on each tool call.

## Known limits
- Only 1 call in dev: Genie answers are trivial until more PDFs are ingested.
- No auth/identity on `session_id`; user-level isolation is not implemented.
