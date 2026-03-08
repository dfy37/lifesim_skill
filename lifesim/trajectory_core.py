"""
Core simulation functions for LifeSim user trajectory generation.

All generation functions use DeepSeek (deepseek-chat) via the OpenAI-compatible API.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

MODEL = "deepseek-chat"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
    return _client


def _call_claude(system: str, user: str, max_tokens: int = 8192) -> str:
    """Call DeepSeek chat, return full text."""
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def _parse_json_response(text: str) -> Any:
    """Extract and parse JSON from a Claude response (strips markdown fences)."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first and last fence lines
        inner = lines[1:-1] if lines[-1].startswith("```") else lines[1:]
        text = "\n".join(inner).strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_life_events(
    user_profile: dict,
    start_date: str,
    end_date: str,
    domains: list[str] | None = None,
) -> list[dict]:
    """
    Generate a timeline of life events for the user.

    Args:
        user_profile: Structured profile of the simulated user.
        start_date: Start of the simulation window (YYYY-MM-DD).
        end_date:   End of the simulation window (YYYY-MM-DD).
        domains:    Life domains to simulate (work, family, health, etc.).

    Returns:
        List of event dicts with keys: event_id, timestamp, domain,
        event_type, description, location, participants, emotional_impact.
    """
    domains = domains or ["work", "family", "health", "entertainment", "travel"]
    system = (
        "You are a life-simulation engine. Generate realistic, coherent life events "
        "for a simulated user based on their profile. Output ONLY valid JSON — a list "
        "of event objects. Each event must have: event_id (string), timestamp "
        "(ISO-8601), domain (string), event_type (string), description (string), "
        "location (string), participants (list of strings), emotional_impact (string)."
    )
    user = (
        f"User profile:\n{json.dumps(user_profile, ensure_ascii=False, indent=2)}\n\n"
        f"Generate life events from {start_date} to {end_date} "
        f"across these domains: {', '.join(domains)}.\n"
        "Produce 15–30 events. Make them temporally ordered, realistic, and "
        "internally consistent with the user's profile. Output JSON only."
    )
    raw = _call_claude(system, user)
    return _parse_json_response(raw)


def generate_travel_trajectory(
    user_profile: dict,
    events: list[dict],
    city: str | None = None,
    granularity: str = "daily",
) -> list[dict]:
    """
    Generate a mobility / travel trajectory based on life events.

    Args:
        user_profile:  Structured user profile.
        events:        Life events from generate_life_events().
        city:          Primary city / geographic anchor.
        granularity:   'daily' or 'hourly'.

    Returns:
        List of trajectory dicts matching the schema in SKILL.md.
    """
    city = city or user_profile.get("city", "Beijing")
    system = (
        "You are a mobility simulation engine. Based on a user profile and their "
        "life events, generate a realistic travel trajectory. "
        "Output ONLY valid JSON — a list of trajectory items. "
        "Each item must have: timestamp (ISO-8601), location (string), "
        "activity (string), transport_mode (string or null), "
        "duration (integer minutes), motivation (string)."
    )
    user = (
        f"User profile:\n{json.dumps(user_profile, ensure_ascii=False, indent=2)}\n\n"
        f"Life events:\n{json.dumps(events, ensure_ascii=False, indent=2)}\n\n"
        f"Primary city: {city}. Granularity: {granularity}.\n"
        "Generate a mobility trajectory consistent with the events above. "
        "Include commutes, errands, social activities, and trips mentioned in events. "
        "Output JSON only."
    )
    raw = _call_claude(system, user, max_tokens=8192)
    return _parse_json_response(raw)


def generate_dialogue_history(
    user_profile: dict,
    events: list[dict] | None = None,
    beliefs: list[dict] | None = None,
    turns: int | None = None,
) -> list[dict]:
    """
    Generate simulated dialogue history between the user and an assistant.

    Args:
        user_profile: Structured user profile.
        events:       Life events (optional, provides grounding).
        beliefs:      Beliefs (optional, enriches dialogue intent).
        turns:        Approximate number of dialogue turns to generate.

    Returns:
        List of dialogue turn dicts matching the schema in SKILL.md.
    """
    turns = turns or 20
    context_parts = [
        f"User profile:\n{json.dumps(user_profile, ensure_ascii=False, indent=2)}"
    ]
    if events:
        context_parts.append(
            f"Recent life events:\n{json.dumps(events[:10], ensure_ascii=False, indent=2)}"
        )
    if beliefs:
        context_parts.append(
            f"User beliefs:\n{json.dumps(beliefs[:5], ensure_ascii=False, indent=2)}"
        )

    system = (
        "You are a dialogue simulation engine. Generate realistic historical "
        "conversations between a user and an AI assistant. "
        "Conversations should reflect the user's life context, needs, and personality. "
        "Output ONLY valid JSON — a list of dialogue turn objects. "
        "Each turn must have: timestamp (ISO-8601), speaker ('user' or 'assistant'), "
        "utterance (string), intent (string), emotion (string or null), "
        "related_event_ids (list), related_belief_ids (list)."
    )
    user = (
        "\n\n".join(context_parts)
        + f"\n\nGenerate approximately {turns} dialogue turns. "
        "Turns must alternate user/assistant. Topics should emerge naturally from "
        "life events and user needs. Output JSON only."
    )
    raw = _call_claude(system, user, max_tokens=8192)
    return _parse_json_response(raw)


def generate_beliefs(
    user_profile: dict,
    events: list[dict] | None = None,
    dialogues: list[dict] | None = None,
    prior_beliefs: list[dict] | None = None,
) -> list[dict]:
    """
    Generate structured belief / state information for the user.

    Args:
        user_profile:   Structured user profile.
        events:         Life events (provides evidence for beliefs).
        dialogues:      Dialogue history (additional evidence).
        prior_beliefs:  Existing beliefs to update (optional).

    Returns:
        List of belief dicts matching the schema in SKILL.md.
    """
    context_parts = [
        f"User profile:\n{json.dumps(user_profile, ensure_ascii=False, indent=2)}"
    ]
    if events:
        context_parts.append(
            f"Life events:\n{json.dumps(events, ensure_ascii=False, indent=2)}"
        )
    if dialogues:
        context_parts.append(
            f"Dialogue history:\n{json.dumps(dialogues[:10], ensure_ascii=False, indent=2)}"
        )
    if prior_beliefs:
        context_parts.append(
            f"Prior beliefs to update:\n{json.dumps(prior_beliefs, ensure_ascii=False, indent=2)}"
        )

    system = (
        "You are a belief-state inference engine. Based on a user's profile, life "
        "events, and dialogue history, extract and structure their beliefs and states. "
        "Output ONLY valid JSON — a list of belief objects. "
        "Each belief must have: belief_id (string), triple ([source, relation, target]), "
        "description (string), belief_type ('stable'|'temporary'|'situational'), "
        "time (YYYY-MM-DD), source_evidence (list of IDs), "
        "confidence (float 0-1), salience (float 0-1)."
    )
    user = (
        "\n\n".join(context_parts)
        + "\n\nInfer 10–20 beliefs. Include both stable personality traits and "
        "dynamic situational states. Each belief must be grounded in evidence. "
        "Output JSON only."
    )
    raw = _call_claude(system, user)
    return _parse_json_response(raw)


def validate_life_consistency(
    user_profile: dict,
    events: list[dict],
    trajectory: list[dict],
    dialogues: list[dict],
    beliefs: list[dict],
) -> dict:
    """
    Check consistency across all generated components.

    Returns:
        Consistency report dict with keys: overall_score, issues, recommendations.
    """
    summary = {
        "user_profile": user_profile,
        "event_count": len(events),
        "trajectory_count": len(trajectory),
        "dialogue_count": len(dialogues),
        "belief_count": len(beliefs),
        "sample_events": events[:3],
        "sample_trajectory": trajectory[:3],
        "sample_dialogues": dialogues[:3],
        "sample_beliefs": beliefs[:3],
    }
    system = (
        "You are a quality-assurance engine for life simulations. "
        "Evaluate temporal, spatial, and persona consistency across all components. "
        "Output ONLY valid JSON with keys: overall_score (float 0-1), "
        "temporal_consistency (float 0-1), spatial_consistency (float 0-1), "
        "persona_consistency (float 0-1), issues (list of strings), "
        "recommendations (list of strings)."
    )
    user = (
        f"Evaluate the following life simulation package:\n"
        f"{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
        "Check for: timeline contradictions, impossible travel, personality inconsistencies, "
        "belief-event mismatches, and dialogue tone mismatches. Output JSON only."
    )
    raw = _call_claude(system, user)
    return _parse_json_response(raw)


def merge_user_context(
    events: list[dict],
    trajectory: list[dict],
    dialogues: list[dict],
    beliefs: list[dict],
) -> dict:
    """
    Merge all generated components into a unified user context package.

    Returns:
        Unified package dict ready for downstream agent use.
    """
    return {
        "schema_version": "1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "life_events": events,
        "travel_trajectory": trajectory,
        "dialogue_history": dialogues,
        "beliefs": beliefs,
        "statistics": {
            "event_count": len(events),
            "trajectory_points": len(trajectory),
            "dialogue_turns": len(dialogues),
            "belief_count": len(beliefs),
        },
    }


def generate_user_life_trajectory(
    user_profile: dict,
    start_date: str,
    end_date: str,
    city: str | None = None,
    domains: list[str] | None = None,
    trajectory_granularity: str = "daily",
    dialogue_density: int = 20,
    belief_mode: str = "mixed",
    output_dir: str = "lifesim_results",
    seed: int | None = None,
) -> dict:
    """
    Complete orchestration pipeline: generate all life trajectory components.

    This is the recommended entry point for standard LifeSim generation.
    It runs all pipeline steps in order and saves all artifacts to disk.

    Args:
        user_profile:             Structured profile of the simulated user.
        start_date:               Simulation start date (YYYY-MM-DD).
        end_date:                 Simulation end date (YYYY-MM-DD).
        city:                     Primary city / geographic anchor.
        domains:                  Life domains to simulate.
        trajectory_granularity:   'daily' or 'hourly'.
        dialogue_density:         Approximate number of dialogue turns.
        belief_mode:              'stable', 'dynamic', or 'mixed'.
        output_dir:               Directory to save all output JSON files.
        seed:                     Reserved for future reproducibility support.

    Returns:
        Unified life trajectory package dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    city = city or user_profile.get("city", "Beijing")

    print("[LifeSim] Step 1/6 — Generating life events...")
    events = generate_life_events(user_profile, start_date, end_date, domains)
    _save(out / "user_life_events.json", events)

    print("[LifeSim] Step 2/6 — Generating travel trajectory...")
    trajectory = generate_travel_trajectory(
        user_profile, events, city=city, granularity=trajectory_granularity
    )
    _save(out / "user_travel_trajectory.json", trajectory)

    print("[LifeSim] Step 3/6 — Generating dialogue history...")
    dialogues = generate_dialogue_history(
        user_profile, events=events, turns=dialogue_density
    )
    _save(out / "user_dialogue_history.json", dialogues)

    print("[LifeSim] Step 4/6 — Generating beliefs...")
    beliefs = generate_beliefs(
        user_profile, events=events, dialogues=dialogues
    )
    _save(out / "user_beliefs.json", beliefs)

    print("[LifeSim] Step 5/6 — Running consistency check...")
    report = validate_life_consistency(
        user_profile, events, trajectory, dialogues, beliefs
    )
    _save(out / "consistency_report.json", report)

    print("[LifeSim] Step 6/6 — Merging and exporting unified package...")
    package = merge_user_context(events, trajectory, dialogues, beliefs)
    package["user_profile"] = user_profile
    package["consistency_report"] = report
    _save(out / "user_life_summary.json", package)

    print(f"[LifeSim] Done. All artifacts saved to: {out.resolve()}")
    return package


def _save(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
