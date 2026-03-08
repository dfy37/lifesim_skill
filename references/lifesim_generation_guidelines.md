# Life Event Generation Guidelines (FastMCP)

This document describes how to use backend tool `generate_life_events` in `lifesim/fastmcp_server.py`.

## Purpose

Generate life events and user intents for each trajectory node from:

- user profile
- sequence id
- expected time horizon

Core implementation references:

- `engine/event_engine.py::OnlineLifeEventEngine`
- especially `generate_event(...)`

## Input Contract

Required:

- `sequence_id: str`
- `user_profile: dict`
- `expected_hours: float`

Optional:

- `start_event_index: int = 0`
- `max_events: int = 8`
- `history_events: list[dict] | None = None`
- `goal: str = ""`

## Output Contract

The returned `nodes` list is the stable frontend-facing structure:

- `event_index`
- `time`
- `location`
- `life_event`
- `intent`
- `sub_intents`
- `weather`

Server also returns metadata:

- `generated_events`
- `next_event_index`
- `requested_events`
- `theme`

## Generation Logic

1. Load sequence by `sequence_id` from `paths.events_path`.
2. Build/obtain retriever index for current theme.
3. Convert profile dict into profile text (`UserProfile.from_dict` if possible).
4. For each node:
   - call `OnlineLifeEventEngine.generate_event(...)`
   - use history as context
   - produce rewritten `life_event` and `intent` when candidates are available.

## Server-Side Config

All infra params are from YAML (not API payload):

- `paths.events_path`
- `paths.event_pool_cfg_path` (optional)
- `models.user_model.*`
- `retriever.embedding_model_name`
- `retriever.persist_directory`
- `retriever.device`

## Operational Notes

- The service caches model/retriever instances in process memory.
- `expected_hours` is mapped to event count by server heuristic (about one event per 3 hours).
- If no remaining event points exist for `start_event_index`, it returns empty `nodes` with message.
