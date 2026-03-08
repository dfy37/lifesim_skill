# LifeSim Generation Guidelines

This document describes the methodology and design principles behind the LifeSim user trajectory generation pipeline.

## Overview

LifeSim generates synthetic but coherent personal histories for simulated users. The pipeline produces four interconnected artefacts — life events, travel trajectories, dialogue history, and belief states — that together form a complete user context package suitable for downstream agent evaluation, personalization research, and long-horizon simulation benchmarking.

## User Profile Schema

A user profile is a JSON object that anchors all generated content. Recommended fields:

```json
{
  "user_id": "u_001",
  "name": "Li Wei",
  "age": 32,
  "gender": "male",
  "occupation": "software engineer",
  "city": "Shanghai",
  "family_status": "married, one child",
  "education": "BSc Computer Science",
  "personality_traits": ["introverted", "analytical", "health-conscious"],
  "hobbies": ["cycling", "cooking", "reading sci-fi"],
  "goals": ["career advancement", "improve fitness", "spend more time with family"],
  "constraints": ["long commute", "limited budget for travel"],
  "health_conditions": [],
  "language": "zh-CN"
}
```

Fields can be extended or omitted; the generation model handles missing fields gracefully with realistic defaults.

## Event-Driven Architecture

All pipeline steps are event-driven:

1. **Life events** are generated first as the ground truth timeline.
2. **Travel trajectories** are derived from events (where did the user go for each event?).
3. **Dialogue history** is grounded in events (what did the user ask an assistant about?).
4. **Beliefs** are inferred from events and dialogues (what patterns emerged?).

This ordering ensures internal consistency: every trajectory segment, dialogue turn, and belief should trace back to at least one life event.

## Temporal Consistency Rules

- All timestamps must fall within the configured `start_date` / `end_date` window.
- Events must be ordered chronologically.
- Travel segments must connect logically (no teleportation; allow transport transitions).
- Dialogue timestamps must correspond to plausible user availability (avoid 3 AM work meetings).
- Belief timestamps must be no earlier than their earliest supporting evidence.

## Realism Guidelines

- **Work events**: Match occupation and schedule (e.g., a teacher has summer holidays; a startup founder works weekends).
- **Family events**: Respect family_status (a single person has different social patterns than a parent).
- **Health events**: If health_conditions is non-empty, include related medical appointments and behavioural patterns.
- **Seasonal variation**: Account for local climate and cultural calendar (Chinese New Year, summer vacation, etc.).
- **Geographic plausibility**: City infrastructure, transport modes, and POI types must match the specified city.

## Pipeline Configuration Tips

| Parameter | Recommended value | Notes |
|---|---|---|
| `trajectory_granularity` | `"daily"` | `"hourly"` increases detail but also token cost |
| `dialogue_density` | 15–30 turns | Fewer turns → less noisy; more → richer context |
| `belief_mode` | `"mixed"` | Combines stable traits with situational states |
| `domains` | default set | Add `"finance"` or `"religion"` for specialist scenarios |

## Quality Checks

After generation, run `validate_life_consistency()` and inspect `consistency_report.json`:

- `overall_score >= 0.75`: Good quality.
- `overall_score < 0.50`: Consider regenerating with a more detailed profile.
- Review `issues` list for specific contradictions to fix manually.

## Known Limitations

- The model may occasionally generate events that are culturally plausible but factually specific (e.g., referencing a real restaurant). Treat all outputs as synthetic.
- Very short time windows (< 7 days) may produce sparse event timelines.
- Highly unusual profiles (e.g., nomadic lifestyle, extreme profession) may require extra prompt guidance via `domains` or profile narrative fields.
