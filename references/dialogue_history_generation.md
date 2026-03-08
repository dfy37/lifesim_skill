# Dialogue History Generation Guidelines

This document explains how LifeSim generates realistic historical conversations between the simulated user and an AI assistant.

## Purpose of Dialogue History

Dialogue history serves as a rich source of implicit user signals. It captures:

- **Expressed needs** — what the user explicitly asked for.
- **Emotional context** — tone and state at the time of interaction.
- **Revealed preferences** — what the user accepted, rejected, or modified.
- **Implicit beliefs** — assumptions embedded in how the user framed requests.

In downstream tasks, dialogue history can be used to warm-start a personalized assistant, construct memory traces, or evaluate whether a model correctly infers user preferences from conversation.

## Dialogue Schema

```json
{
  "timestamp": "2025-02-12T21:03:00",
  "speaker": "user",
  "utterance": "I'm exhausted after commuting today. Can you help me find a quieter route tomorrow?",
  "intent": "seek_commute_optimization",
  "emotion": "tired",
  "related_event_ids": ["event_023"],
  "related_belief_ids": ["belief_011"]
}
```

### Field Definitions

- **timestamp**: ISO-8601 datetime of the utterance.
- **speaker**: `"user"` or `"assistant"`.
- **utterance**: Natural-language text of the turn.
- **intent**: High-level communicative purpose (snake_case). Common values: `seek_information`, `make_request`, `confirm`, `reject`, `clarify`, `express_emotion`, `seek_recommendation`, `plan_activity`, `report_event`.
- **emotion**: Emotional tone of the utterance (may be null for assistant turns). Common values: `happy`, `tired`, `anxious`, `excited`, `neutral`, `frustrated`, `grateful`.
- **related_event_ids**: IDs of life events that motivated or are referenced by this turn.
- **related_belief_ids**: IDs of beliefs reflected in this turn (may be empty).

## Design Principles

### 1. Ground Dialogue in Events

Each user turn should be traceable to at least one life event. A user who just had a stressful meeting is more likely to ask for relaxation suggestions; a user who booked a trip is more likely to ask about travel logistics.

### 2. Maintain Natural Turn Flow

- User turns express needs, ask questions, or provide feedback.
- Assistant turns answer, suggest, confirm, or ask clarifying questions.
- Avoid isolated single-turn exchanges — most topics span 2–4 turns.

### 3. Vary Topics and Register

Good dialogue history covers diverse topics (work, family, health, entertainment) and registers (casual chat, precise information requests, emotional support). A monotone history weakens personalization signal.

### 4. Embed Implicit Preference Signals

Some of the most valuable personalization signal comes from *what the user did not say* explicitly:

- A user who always asks for vegetarian restaurants reveals a dietary preference.
- A user who frequently asks about traffic before departure reveals a planning habit.
- A user who dismisses the first suggestion and accepts the second reveals a selectiveness trait.

### 5. Time-Align with Life Events

Dialogue timestamps should cluster around event timestamps:

- Pre-event: planning, booking, asking for advice.
- Post-event: reporting outcomes, expressing reactions, seeking follow-up.
- Mid-day gaps: represent periods when the user was unavailable.

## Dialogue Density Guidelines

| Density | Turns | Best for |
|---|---|---|
| Low | 8–12 | Quick prototyping, low-noise contexts |
| Medium | 15–25 | Standard evaluation benchmarks |
| High | 30–50 | Rich personalization and memory tasks |

High-density histories increase token usage significantly. Use `dialogue_density=20` as the default.

## Common Failure Modes to Avoid

- **Repetitive topics**: Every pair of turns discusses the same theme — ensure variety across the full history.
- **Overly formal language**: Real user messages are casual and often fragmentary. Vary register.
- **Unanchored emotions**: Emotions should match the surrounding life context, not be randomly assigned.
- **Missing assistant acknowledgements**: The assistant should reference what the user said, not give generic responses.
- **Future-dating**: All dialogue timestamps must fall within the simulation window and before the current generation date.
