---
name: lifesim-user-trajectory-generation
description: Generates a complete user life trajectory package for a given user profile, including travel trajectory, dialogue history, and belief states. Use when users request long-horizon user simulation, personalized life-context generation, dynamic user state construction, or synthetic history generation for agent personalization and evaluation.
trigger: Use this skill when users ask to generate a user life trajectory, simulate user history, create a LifeSim package, or generate synthetic personal context for agent evaluation.
requirements:
  - python: ">=3.10"
  - install: "pip install -r requirements.txt"
  - env: "ANTHROPIC_API_KEY must be set"
---

# LifeSim User Trajectory Generation

Automated long-horizon user life simulation workflow for generating structured personal history, mobility patterns, and cognitive state information.

## Setup

```bash
# Install dependencies (requires Python ≥ 3.10)
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Quick Start

```python
from lifesim import generate_user_life_trajectory

user_profile = {
    "user_id": "u_001",
    "name": "Li Wei",
    "age": 32,
    "occupation": "software engineer",
    "city": "Shanghai",
    "personality_traits": ["introverted", "analytical", "health-conscious"],
    "hobbies": ["cycling", "cooking", "reading sci-fi"],
    "goals": ["career advancement", "improve fitness"],
}

result = generate_user_life_trajectory(
    user_profile=user_profile,
    start_date="2025-01-01",
    end_date="2025-03-31",
    city="Shanghai",
    output_dir="lifesim_results",
)

print(f"Generated {len(result['life_events'])} events")
print(f"Consistency score: {result['consistency_report']['overall_score']:.2f}")
```

## When to Use This Skill

Use when users:
- Request generation of a complete user life trajectory
- Want synthetic user history for personalized agent evaluation
- Need travel/mobility trajectories for a simulated user
- Need dialogue history between the user and an assistant
- Need belief/state information inferred from life events and interactions
- Ask to construct long-horizon user context for memory, planning, or personalization
- Want a full LifeSim package for downstream agent reasoning or benchmarking

**Supported input formats:**
- Structured user profile in JSON / dict format
- Optional persona description in natural language
- Optional scenario constraints (time range, city, domain, life events, goals)

**Default recommendation**: Use Approach 1 (complete trajectory generation pipeline) unless the user explicitly requests only one component (e.g., only belief generation or only dialogue history).

## Approach 1: Complete User Trajectory Generation Pipeline (Recommended for Standard Workflows)

For standard LifeSim generation, use the main orchestration workflow to generate all three components together:

```python
from lifesim import generate_user_life_trajectory

result = generate_user_life_trajectory(
    user_profile=user_profile,
    start_date="2025-01-01",
    end_date="2025-03-31",
    city="Shanghai",
    output_dir="lifesim_results"
)
````

**When to use this approach:**

* Standard long-horizon user simulation
* Building a full user context package for agent personalization
* Benchmark construction for personalized assistants
* Generating synthetic but coherent personal history
* Quick end-to-end generation with minimal configuration

**Core outputs:**

* Travel trajectory
* Dialogue history
* Belief information
* Optional event timeline and consistency report

**Parameters:**

Customize generation behavior using the following parameters:

* `user_profile` - Structured profile of the simulated user
* `start_date`, `end_date` - Time window for life trajectory generation
* `city` - Main living environment or geographic anchor
* `domains` - Domains to simulate (e.g., work, family, health, entertainment, travel)
* `trajectory_granularity` - Temporal granularity of movement/events (daily / hourly)
* `dialogue_density` - Approximate amount of dialogue history to generate
* `belief_mode` - Belief extraction style (stable / dynamic / mixed)
* `output_dir` - Directory for saving generated artifacts
* `seed` - Random seed for reproducibility

**Outputs:**

All files are saved to `lifesim_results/` by default (or to the directory specified by `--output-dir`):

* `user_travel_trajectory.json` - Structured mobility and travel trajectory
* `user_dialogue_history.json` - Simulated dialogue history between user and assistant
* `user_beliefs.json` - Structured belief/state information with temporal annotations
* `user_life_events.json` - Generated life event timeline
* `user_life_summary.json` - Unified summary package for downstream use
* `consistency_report.json` - Optional report on temporal/persona consistency

If copying outputs for user access, copy individual files rather than the whole directory so that each artifact can be directly previewed or loaded downstream.

### Workflow Steps

The pipeline performs the following steps:

1. **Load and normalize user profile** - Parse structured profile fields, goals, preferences, and constraints
2. **Generate life events** - Simulate key events over the target time window
3. **Generate travel trajectory** - Produce movement and location sequences based on events and routines
4. **Generate dialogue history** - Simulate prior user-assistant conversations grounded in events and needs
5. **Generate beliefs** - Infer stable and dynamic belief states from events and dialogues
6. **Run consistency checking** - Verify temporal, spatial, and persona coherence
7. **Export unified trajectory package** - Save all artifacts in structured machine-readable form

## Approach 2: Modular Building Blocks (For Custom Workflows)

For custom simulation workflows or non-standard requirements, use the modular generators independently:

```python
from lifesim import (
    generate_life_events,
    generate_travel_trajectory,
    generate_dialogue_history,
    generate_beliefs,
    validate_life_consistency
)

events = generate_life_events(user_profile, start_date="2025-01-01", end_date="2025-03-31")
trajectory = generate_travel_trajectory(user_profile, events=events)
dialogues = generate_dialogue_history(user_profile, events=events)
beliefs = generate_beliefs(user_profile, events=events, dialogues=dialogues)
report = validate_life_consistency(
    user_profile=user_profile,
    events=events,
    trajectory=trajectory,
    dialogues=dialogues,
    beliefs=beliefs
)
```

**When to use this approach:**

* Only one component needs to be generated
* Different logic is required for different submodules
* The user wants partial execution
* The pipeline must be integrated into a larger simulation framework
* Custom filtering, editing, or post-processing is needed
* Different generation strategies are needed for different user subgroups

**Available utility functions:**

From `trajectory_core.py` (core simulation operations):

* `generate_life_events(user_profile, start_date, end_date, domains=None)` - Generate event timeline
* `generate_travel_trajectory(user_profile, events, city=None, granularity='daily')` - Generate location and movement history
* `generate_dialogue_history(user_profile, events=None, beliefs=None, turns=None)` - Generate historical conversations
* `generate_beliefs(user_profile, events=None, dialogues=None, prior_beliefs=None)` - Generate belief/state information
* `merge_user_context(events, trajectory, dialogues, beliefs)` - Merge all generated modules into a unified package
* `validate_life_consistency(user_profile, events, trajectory, dialogues, beliefs)` - Check consistency across outputs

From `trajectory_utils.py` (analysis and export helpers):

* `export_json(data, output_path)` - Save structured outputs
* `summarize_life_package(package)` - Produce a compact natural-language summary
* `sample_recent_context(package, k=5)` - Retrieve most recent events/dialogues/beliefs
* `filter_by_time_range(data, start_date, end_date)` - Slice generated history by time

**Example custom workflows:**

**Example 1: Only generate dialogue history**

```python
dialogues = generate_dialogue_history(
    user_profile=user_profile,
    turns=20
)
```

**Example 2: Generate travel trajectory from existing events**

```python
events = generate_life_events(
    user_profile=user_profile,
    start_date="2025-01-01",
    end_date="2025-01-31"
)

trajectory = generate_travel_trajectory(
    user_profile=user_profile,
    events=events,
    city="Beijing",
    granularity="hourly"
)
```

**Example 3: Generate beliefs based only on prior events and dialogues**

```python
beliefs = generate_beliefs(
    user_profile=user_profile,
    events=events,
    dialogues=dialogues,
    prior_beliefs=None
)
```

**Example 4: Build a compact package for current assistant reasoning**

```python
package = merge_user_context(
    events=events,
    trajectory=trajectory,
    dialogues=dialogues,
    beliefs=beliefs
)

recent_context = sample_recent_context(package, k=10)
```

## Generated Data Schema

### Travel Trajectory Schema

Each trajectory item should include:

* `timestamp` - Time of movement or stay
* `location` - Place name or semantic location
* `activity` - Activity associated with the location
* `transport_mode` - Transportation mode if applicable
* `duration` - Duration of stay or trip segment
* `motivation` - Why the user went there

Example:

```json
{
  "timestamp": "2025-02-12T08:10:00",
  "location": "Office District",
  "activity": "commuting to work",
  "transport_mode": "metro",
  "duration": 45,
  "motivation": "weekday work routine"
}
```

### Dialogue History Schema

Each dialogue turn should include:

* `timestamp` - Time of the utterance
* `speaker` - `user` or `assistant`
* `utterance` - Natural language utterance
* `intent` - High-level purpose of the utterance
* `emotion` - Optional emotional tone
* `related_event_ids` - Referenced life events
* `related_belief_ids` - Referenced beliefs

Example:

```json
{
  "timestamp": "2025-02-12T21:03:00",
  "speaker": "user",
  "utterance": "I’m exhausted after commuting today. Can you help me find a quieter route tomorrow?",
  "intent": "seek commute optimization",
  "emotion": "tired",
  "related_event_ids": ["event_023"],
  "related_belief_ids": ["belief_011"]
}
```

### Belief Information Schema

Each belief item should include:

* `belief_id` - Unique identifier
* `triple` - Structured belief triple `[source, relation, target]`
* `description` - Natural language explanation
* `belief_type` - Stable / temporary / situational
* `time` - Timestamp or validity start time
* `source_evidence` - Supporting events or dialogues
* `confidence` - Confidence score
* `salience` - Importance score

Example:

```json
{
  "belief_id": "belief_011",
  "triple": ["user", "prefers", "quiet_commute"],
  "description": "The user has recently developed a stronger preference for less crowded commuting routes.",
  "belief_type": "situational",
  "time": "2025-02-12",
  "source_evidence": ["event_023", "dialogue_054"],
  "confidence": 0.84,
  "salience": 0.79
}
```

## Best Practices

1. **Generate events before beliefs** - Beliefs should be grounded in events and interactions rather than invented in isolation
2. **Maintain temporal consistency** - Travel, dialogue, and beliefs must align with the same timeline
3. **Keep beliefs interpretable** - Use structured triples and natural-language descriptions together
4. **Use realistic dialogue density** - Too much dialogue makes the package noisy; too little weakens personalization context
5. **Preserve persona coherence** - Generated behavior should remain consistent with profile traits, routines, and constraints
6. **Allow dynamic belief change** - Some beliefs should evolve over time as events accumulate
7. **Inspect outputs manually when needed** - Especially for long-horizon settings or benchmark release data

## Implementation Notes

- All generation functions call `claude-opus-4-6` with **streaming** and **adaptive thinking** (`thinking: {type: "adaptive"}`).
- The `ANTHROPIC_API_KEY` environment variable must be set before calling any function.
- Long generations (full pipeline, hourly granularity) can take 2–5 minutes due to multi-step Claude calls.
- All outputs are saved as UTF-8 JSON files; existing files are overwritten on re-run.
- The `_client` singleton in `trajectory_core.py` is reused across calls within a Python session.

## Reference Materials

For detailed methodology, data schema design, and generation rationale, see:

* `references/lifesim_generation_guidelines.md`
* `references/belief_modeling_guidelines.md`
* `references/dialogue_history_generation.md`

These references may provide:

* Detailed explanation of event-driven user simulation
* How beliefs are derived from profile, events, and dialogue
* Temporal consistency rules across generated history
* Recommendations for travel/mobility realism
* Guidelines for generating assistant-facing historical conversations
* Suggestions for adapting the pipeline to different domains

Load these references when users need a deeper understanding of the simulation methodology or when troubleshooting generation quality.

## Next Steps After Generation

Typical downstream uses:

* Personalized assistant evaluation
* Memory-augmented agent testing
* User state reconstruction experiments
* Implicit intention inference benchmarking
* Long-horizon dialogue simulation
* Planning and recommendation evaluation
* Persona alignment and profile recovery assessment