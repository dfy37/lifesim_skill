"""
LifeSim: Long-horizon user life trajectory generation using Claude AI.
"""

from .trajectory_core import (
    generate_user_life_trajectory,
    generate_life_events,
    generate_travel_trajectory,
    generate_dialogue_history,
    generate_beliefs,
    merge_user_context,
    validate_life_consistency,
)
from .trajectory_utils import (
    export_json,
    summarize_life_package,
    sample_recent_context,
    filter_by_time_range,
)

__all__ = [
    "generate_user_life_trajectory",
    "generate_life_events",
    "generate_travel_trajectory",
    "generate_dialogue_history",
    "generate_beliefs",
    "merge_user_context",
    "validate_life_consistency",
    "export_json",
    "summarize_life_package",
    "sample_recent_context",
    "filter_by_time_range",
]
