# Dialogue Generation Guidelines (FastMCP)

This document describes backend tool `generate_event_dialogues` in `lifesim/fastmcp_server.py`.

## Purpose

Generate event-grounded dialogue history between user and assistant.

Each event experience becomes one dialogue block.

## Input Contract

Required:

- `user_profile: dict`
- `event_experiences: list[dict]`

Optional:

- `beliefs: list | None = None`
- `max_turns: int = 6`
- `refine_intention_enabled: bool = true`

## Event Fields Expected

For each event item, at least one of these should exist:

- `life_event` or `event`

For intention extraction (priority order):

- `intent`
- `intention`
- `user_intention`

If none provided, server uses a fallback intention based on event text.

## Output Contract

Returns:

- `generated_dialogues`
- `max_turns`
- `dialogues[]`

Each `dialogues[]` item:

- `event_index`
- `event`
- `intention`
- `dialogue` (list of turns)

Turn format:

```json
{ "role": "user|assistant", "content": "..." }
```

## Internal Construction

The tool uses:

- `FastConvSimulator.simulate(...)` from `simulation/fast_conv_simulator.py`
- optional intention rewrite via `refine_intention(...)` from `simulation/conv_history_generator.py`

Flow per event:

1. Parse event text and intention.
2. Optionally refine intention with profile + beliefs + event context.
3. Generate multi-turn dialogue with max turn constraint.
4. Return structured turns.

## Recommended Pairing With Event Tool

Typical backend orchestration:

1. `generate_life_events`
2. `generate_event_dialogues` using generated nodes/events

This keeps event-intention-dialogue consistent in one request chain.
