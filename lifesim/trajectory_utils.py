"""
Utility functions for LifeSim: export, analysis, and context helpers.
"""

import json
from pathlib import Path
from typing import Any

import anthropic

MODEL = "claude-opus-4-6"


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def export_json(data: Any, output_path: str) -> None:
    """
    Save structured data to a JSON file.

    Args:
        data:        Any JSON-serialisable object.
        output_path: File path to write (created if it doesn't exist).
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def summarize_life_package(package: dict) -> str:
    """
    Produce a compact natural-language summary of the life package.

    Uses Claude to generate a concise narrative summary.

    Args:
        package: Unified package from merge_user_context() or
                 generate_user_life_trajectory().

    Returns:
        Natural-language summary string.
    """
    client = anthropic.Anthropic()
    stats = package.get("statistics", {})
    profile = package.get("user_profile", {})
    sample_events = package.get("life_events", [])[:5]
    sample_beliefs = package.get("beliefs", [])[:3]

    prompt_data = {
        "user_profile": profile,
        "statistics": stats,
        "sample_events": sample_events,
        "sample_beliefs": sample_beliefs,
    }

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=(
            "You are a life-simulation analyst. Write a concise 2–3 paragraph "
            "narrative summary of the user's simulated life context. "
            "Focus on key patterns, notable events, and dominant belief states."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Summarize this life package:\n"
                    f"{json.dumps(prompt_data, ensure_ascii=False, indent=2)}"
                ),
            }
        ],
    )
    return response.content[0].text


def sample_recent_context(package: dict, k: int = 5) -> dict:
    """
    Retrieve the k most recent events, dialogues, and beliefs.

    Args:
        package: Unified package from merge_user_context().
        k:       Number of items to retrieve per category.

    Returns:
        Dict with keys: recent_events, recent_dialogues, recent_beliefs.
    """
    events = package.get("life_events", [])
    dialogues = package.get("dialogue_history", [])
    beliefs = package.get("beliefs", [])

    def _sort_by_time(items: list[dict]) -> list[dict]:
        return sorted(
            items,
            key=lambda x: x.get("timestamp", x.get("time", "")),
            reverse=True,
        )

    return {
        "recent_events": _sort_by_time(events)[:k],
        "recent_dialogues": _sort_by_time(dialogues)[:k],
        "recent_beliefs": beliefs[-k:],  # keep last k (assumed ordered)
    }


def filter_by_time_range(
    data: list[dict], start_date: str, end_date: str
) -> list[dict]:
    """
    Slice a list of timestamped dicts to a specific date range.

    Args:
        data:       List of dicts with a 'timestamp' or 'time' key.
        start_date: Inclusive start (YYYY-MM-DD or ISO-8601).
        end_date:   Inclusive end (YYYY-MM-DD or ISO-8601).

    Returns:
        Filtered list ordered by timestamp.
    """
    result = []
    for item in data:
        ts = item.get("timestamp") or item.get("time", "")
        date_part = ts[:10]  # YYYY-MM-DD
        if start_date <= date_part <= end_date:
            result.append(item)
    return sorted(result, key=lambda x: x.get("timestamp", x.get("time", "")))
