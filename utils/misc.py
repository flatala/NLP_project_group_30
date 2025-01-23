"""Miscellaneous utility functions for the project."""

import json
import string
from .prompts import SYSTEM_PROMPT, EXAMPLE_USER_PROMPT, EXAMPLE_ASSISTANT_RESPONSE


def normalize_string(text: str) -> str:
    text = text.strip().lower()
    text = text.strip(string.punctuation)
    return " ".join(text.split())


def exact_match_score(predicted: str, ground_truth: str) -> float:
    if predicted is not None:
        normalized_predicted = normalize_string(predicted)
        normalized_ground_truth = normalize_string(ground_truth)
        return 1.0 if normalized_predicted == normalized_ground_truth else 0.0
    else:
        return 0.0


def cover_exact_match_score(predicted: str, ground_truth: str) -> float:
    if predicted is not None:
        normalized_predicted = normalize_string(predicted)
        normalized_ground_truth = normalize_string(ground_truth)
        if (
            normalized_predicted in normalized_ground_truth
            or normalized_ground_truth in normalized_predicted
        ):
            return 1.0
        else:
            return 0.0
    else:
        return 0.0


def prepare_prompt(question: str, evidences: list[str]) -> list[dict[str, str]]:
    full_prompt: list[dict[str, str]] = []
    full_prompt.append({"role": "system", "content": SYSTEM_PROMPT})
    full_prompt.append({"role": "user", "content": EXAMPLE_USER_PROMPT})
    full_prompt.append({"role": "assistant", "content": EXAMPLE_ASSISTANT_RESPONSE})
    qa_structure = {
        "question": question,
        "evidences": evidences,
    }
    full_prompt.append({"role": "user", "content": json.dumps(qa_structure)})
    return full_prompt


def get_answer_from_model_output(model_output: dict) -> str:
    output = model_output[-1]
    messages = output.get("generated_text")
    final_response = messages[-1]
    answer_struct = final_response.get("content")
    answer_dict = json.loads(answer_struct)
    return answer_dict.get("final_answer")
