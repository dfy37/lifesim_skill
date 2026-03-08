---
name: lifesim-fastmcp-backend
description: Build and use the LifeSim FastMCP backend service. Primary capabilities: (1) generate life-event trajectories with user intents, (2) generate user-assistant dialogues from profile + event experiences.
trigger: Use this skill when users ask to run LifeSim on backend server, expose MCP tools, generate life events/intents, or generate event-grounded dialogue history.
requirements:
  - python: ">=3.10"
  - install: "Install dependencies required by lifesim and mcp server runtime"
  - env: "Frontend/client does not need model-provider key; all model/retriever config is server-side in YAML"
---

# LifeSim FastMCP Backend Skill

This skill is for backend deployment using `lifesim/fastmcp_server.py`.

## Scope

This backend currently provides two MCP tools:

1. `generate_life_events`
2. `generate_event_dialogues`

## Architecture Rule

- All generation logic runs in backend (`lifesim` code + model/retriever init).
- Frontend only calls backend APIs and renders results.
- Do not expose provider keys to frontend.

## Server Entry

Main file:

- `lifesim/fastmcp_server.py`

Start server:

```bash
python lifesim/fastmcp_server.py --config config.yaml --transport streamable-http
```

Supported transport:

- `streamable-http` (recommended)
- `sse`
- `stdio`

Environment variables:

- `LIFESIM_MCP_HOST` (default `0.0.0.0`)
- `LIFESIM_MCP_PORT` (default `8000`)
- `LIFESIM_CONFIG_PATH` (default `config.yaml`)

## Server Config (YAML)

Runtime infra parameters are read from YAML, not from tool input:

```yaml
paths:
  events_path: "/path/to/event_sequences.jsonl"
  event_pool_cfg_path: "/path/to/events_pool_cfgs.json"  # optional

models:
  user_model:
    model_name: "deepseek-chat"  # optional; if omitted, inferred from model_path basename
    model_path: "deepseek-chat"
    api_key: "<server-side-key>"
    base_url: "https://api.deepseek.com"
    vllmapi: true

retriever:
  embedding_model_name: "/path/to/embedding/model"
  persist_directory: "/path/to/chroma_db"
  device: "auto"
```

## Tool 1: `generate_life_events`

Generate event nodes and intent for each node based on user profile and expected time horizon.

Input:

- `sequence_id: str`
- `user_profile: dict`
- `expected_hours: float`
- `start_event_index: int = 0`
- `max_events: int = 8`
- `history_events: list[dict] | None = None`
- `goal: str = ""`

Output (core fields):

- `nodes[]` with:
  - `event_index`
  - `time`
  - `location`
  - `life_event`
  - `intent`
  - `sub_intents`
  - `weather`
- plus metadata (`generated_events`, `next_event_index`, etc.)

Implementation basis:

- `engine/event_engine.py::OnlineLifeEventEngine.generate_event`
- retrieval/rerank/rewrite flow and history conditioning are inherited from engine logic.

## Tool 2: `generate_event_dialogues`

Generate user-assistant dialogue for each experienced event.

Input:

- `user_profile: dict`
- `event_experiences: list[dict]`
- `beliefs: list | None = None`
- `max_turns: int = 6`
- `refine_intention_enabled: bool = true`

Output:

- `dialogues[]` with:
  - `event_index`
  - `event`
  - `intention`
  - `dialogue` (turn list: `{role, content}`)

Implementation basis:

- `simulation/fast_conv_simulator.py::FastConvSimulator.simulate`
- `simulation/conv_history_generator.py::refine_intention`

## Recommended Backend Flow

1. Call `generate_life_events` to get event trajectory + intents.
2. Pass returned `nodes` (or `raw_events`) into `generate_event_dialogues`.
3. Return unified payload to frontend.

## Frontend Integration Pattern

Use a backend API gateway/BFF to map HTTP endpoints to MCP tool calls.

Example endpoint mapping:

- `POST /api/lifesim/generate-life-events` -> MCP `generate_life_events`
- `POST /api/lifesim/generate-event-dialogues` -> MCP `generate_event_dialogues`

Do not call model vendors directly from frontend.

## References

- `references/lifesim_generation_guidelines.md`
- `references/dialogue_history_generation.md`
- `references/belief_modeling_guidelines.md`
- `references/frontend_lifesim_api_demo.js`
