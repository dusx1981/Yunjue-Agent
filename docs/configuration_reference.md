## Configuration reference

This document explains the meaning of the main configuration fields used by this project.

- Environment variables live in `.env` (see `.env.example`).
- Model / LLM settings live in `conf.yaml` (see `conf.yaml.example`).

Keep secrets (API keys) out of git. Never commit real keys.

## `.env` (environment variables)

### `TAVILY_API_KEY`

API key for Tavily web search (used by tools that need high-quality web search).

- How to get it: create an account and generate an API key from `https://app.tavily.com/home`.
- Usage: tools authenticate with a Bearer token (conceptually `Authorization: Bearer {TAVILY_API_KEY}`).

### `MAX_WORKER_RECURSION_LIMIT`

Maximum number of tool-usage “steps” a **ReAct worker agent** can execute for a single task step.

Concretely, this sets the worker agent’s `max_steps` (i.e., the maximum number of times the worker can loop and call tools). If the worker reaches this limit **without producing a final response**, execution is stopped and control returns to the manager, which will analyze the execution trace / partial results and decide what to do next (e.g., retry with a failure report).

### `MAX_TASK_EXECUTION_CNT`

Maximum number of times the **manager** will analyze a step’s execution result and attempt to recover (diagnose + provide suggestions + re-run the worker).

In other words:

- The worker loops up to `MAX_WORKER_RECURSION_LIMIT` tool calls in one run.
- The manager can trigger that worker run again (with guidance) up to `MAX_TASK_EXECUTION_CNT` times for the same step.

### `OPENAI_API_KEY` (Codex-related)

API key used by OpenAI tooling such as the **Codex CLI**. This repository’s Codex integration is designed to work with the external `codex` command, which typically relies on `OPENAI_API_KEY` being present in the environment.

If you are not using Codex-based workflows, you can leave this unset.

### `CODEX_PROFILE` (Codex-related)

Optional profile name passed to the external `codex` CLI (used when running `codex exec ... --profile <CODEX_PROFILE>`).

If you are not using Codex-based workflows, you can leave this unset.

### `PROXY_URL`

Optional proxy URL. When this is set, **all network-related tools created by the tool developer** will route their requests through this proxy.

## `conf.yaml` (LLM / model configuration)

`conf.yaml` contains the model settings for different roles in the system. Each entry is a dictionary passed to the LLM client (for example, `base_url`, `model`, `api_key`, `temperature`, `timeout`, and a project-specific `token_limit`).

Common fields (per model block):

- **`base_url`**: API base URL for your model provider / gateway.
- **`model`**: model name (provider-specific).
- **`api_key`**: API key for that provider.
- **`temperature`**: sampling temperature.
- **`token_limit`**: maximum tokens expected/allowed (used by the project to size contexts).
- **`max_retries`**: retry count for transient failures.
- **`timeout`**: request timeout in seconds.

### `VISION_MODEL`

Model used specifically for **vision / image understanding** tasks (i.e., visual analysis). Pick a multimodal model that supports image inputs.

### `SUMMARIZE_MODEL`

Model used for **summarization when tool outputs are too long**, or when the system needs to compress intermediate context into a “context summary” and extract only the task-relevant information.
