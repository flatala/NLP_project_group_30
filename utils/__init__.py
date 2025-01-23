"""Contains the patched and updated classes for the Notebook"""

from .adore import AdoreRetriever
from .contriever import ContrieverRetriever
from .misc import (
    get_answer_from_model_output,
    normalize_string,
    exact_match_score,
    cover_exact_match_score,
    prepare_prompt,
)
from .plots import plot_accuracy_bar_chart
from .prompts import EXAMPLE_ASSISTANT_RESPONSE, EXAMPLE_USER_PROMPT, SYSTEM_PROMPT


__all__ = [
    "AdoreRetriever",
    "ContrieverRetriever",
    "plot_accuracy_bar_chart",
    "get_answer_from_model_output",
    "normalize_string",
    "exact_match_score",
    "cover_exact_match_score",
    "prepare_prompt",
    "EXAMPLE_ASSISTANT_RESPONSE",
    "EXAMPLE_USER_PROMPT",
    "SYSTEM_PROMPT",
]
