# Belief Modeling Guidelines

This document explains how LifeSim models user beliefs and cognitive states, and how to interpret the belief output schema.

## What Is a Belief?

In LifeSim, a *belief* is a structured representation of something the user holds to be true, values, prefers, or intends. Beliefs are derived from life events and dialogue history; they are never invented without supporting evidence.

Beliefs come in three types:

| Type | Description | Example |
|---|---|---|
| `stable` | Persistent personality traits or deep preferences | "user prefers quiet environments" |
| `temporary` | Short-lived states that arise from specific events | "user is stressed about project deadline" |
| `situational` | Context-dependent preferences or intentions | "user prefers metro over taxi when it rains" |

## Belief Schema

```json
{
  "belief_id": "belief_011",
  "triple": ["user", "prefers", "quiet_commute"],
  "description": "The user has developed a stronger preference for less crowded commuting routes after a series of stressful rush-hour journeys.",
  "belief_type": "situational",
  "time": "2025-02-12",
  "source_evidence": ["event_023", "dialogue_054"],
  "confidence": 0.84,
  "salience": 0.79
}
```

### Field Definitions

- **belief_id**: Unique identifier (format: `belief_NNN`).
- **triple**: `[subject, relation, object]`. Subject is usually `"user"`. Relations include: `prefers`, `dislikes`, `intends`, `believes`, `fears`, `values`, `owns`, `knows`, `plans`.
- **description**: Full natural-language explanation including context and temporal nuance.
- **belief_type**: `stable` / `temporary` / `situational` (see table above).
- **time**: ISO date when this belief first became observable.
- **source_evidence**: IDs from `life_events` or `dialogue_history` that support this belief.
- **confidence**: How certain we are this belief is currently held (0.0–1.0).
- **salience**: How important / foregrounded this belief is for downstream reasoning (0.0–1.0).

## Generating High-Quality Beliefs

1. **Ground every belief in evidence.** Do not generate a belief without at least one supporting event or dialogue turn.
2. **Vary belief types.** A realistic profile has a mix of ~40% stable, ~30% temporary, ~30% situational beliefs.
3. **Update temporal beliefs.** If the same belief appears in multiple events, update `time` to the most recent occurrence and adjust `confidence` accordingly.
4. **Distinguish preference from behaviour.** A user may *prefer* vegetarian food (belief) but eat meat occasionally (event). Both can coexist.
5. **Score confidence conservatively.** Reserve scores above 0.90 for beliefs reinforced by 3+ independent evidence items.

## Using Beliefs in Downstream Tasks

- **Personalized recommendation**: Filter candidates by beliefs with `salience > 0.6`.
- **Implicit intention inference**: Look for `intends` and `plans` triples with recent `time` values.
- **Memory-augmented dialogue**: Inject top-5 salient beliefs into the assistant's context.
- **Persona alignment evaluation**: Check whether generated responses are consistent with stable beliefs.

## Dynamic Belief Updates

When simulating belief change over time (e.g., with `belief_mode = "dynamic"`):

- Create a new belief entry for each update rather than overwriting.
- Decrement `confidence` of the old version to represent decay.
- Set `belief_type = "temporary"` for beliefs that are likely to revert after the triggering event passes.
