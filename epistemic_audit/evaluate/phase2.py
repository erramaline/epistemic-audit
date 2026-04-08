"""Phase 2: Audit AUROC and planted detection."""

import re
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score
from epistemic_audit import AuditRating


@dataclass
class Phase2Results:
    """Stores all Phase 2 evaluation metrics."""

    audit_auroc: float
    planted_detection_rate: float
    n_items: int
    ratings: list[AuditRating]
    actual_correctness: list[bool]


def _try_fallback_parse(item_text: str) -> dict | None:
    """Fallback parser: extract any number after RATING and YES/NO after FLAGGED.

    Args:
        item_text: Raw text for a single item block.

    Returns:
        Parsed dict with 'rating', 'flagged', 'explanation', or None if nothing found.
    """
    rating_match = re.search(r"RATING[:\s]+(\d+)", item_text, re.IGNORECASE)
    flagged_match = re.search(r"FLAGGED[:\s]+(YES|NO)", item_text, re.IGNORECASE)
    expl_match = re.search(r"EXPLANATION[:\s]+(.+)", item_text, re.IGNORECASE | re.DOTALL)

    if rating_match or flagged_match:
        return {
            "rating": max(0, min(100, int(rating_match.group(1)))) if rating_match else 50,
            "flagged": flagged_match.group(1).upper() == "YES" if flagged_match else False,
            "explanation": expl_match.group(1).strip()[:200] if expl_match else "",
        }
    return None


def parse_single_item_response(item_text: str, item_number: int) -> dict:
    """Parse the response for a single audit item from a batch response.

    Args:
        item_text: Raw text containing only this one item's block.
        item_number: The item number (1-indexed) to look for in the text.

    Returns:
        Dict with 'rating' (0-100), 'flagged' (bool), 'explanation' (str).
        Defaults to rating=50, flagged=False if parsing fails.
    """
    pattern = (
        rf"ITEM\s*{item_number}.*?RATING:\s*(\d+).*?FLAGGED:\s*(YES|NO)"
        rf".*?EXPLANATION:\s*(.+?)(?=ITEM\s*\d+|$)"
    )
    match = re.search(pattern, item_text, re.DOTALL | re.IGNORECASE)
    if match:
        return {
            "rating": max(0, min(100, int(match.group(1)))),
            "flagged": match.group(2).upper() == "YES",
            "explanation": match.group(3).strip(),
        }

    fallback = _try_fallback_parse(item_text)
    if fallback:
        return fallback

    return {"rating": 50, "flagged": False, "explanation": ""}


def parse_phase2_response(raw: str, n_items: int) -> list[dict]:
    """Parse a Phase 2 audit response into a list of item ratings.

    Tries strict regex first; falls back to looser number/keyword extraction
    to handle models that deviate slightly from the required format.

    Args:
        raw: Raw model response string containing all item ratings.
        n_items: Total number of audit items expected.

    Returns:
        List of dicts, each with 'rating' (0-100), 'flagged' (bool),
        'explanation' (str). Items that fail parsing default to rating=50.
    """
    items = []
    for i in range(1, n_items + 1):
        pattern = (
            rf"ITEM\s*{i}.*?RATING:\s*(\d+).*?FLAGGED:\s*(YES|NO)"
            rf".*?EXPLANATION:\s*(.+?)(?=ITEM\s*\d+|$)"
        )
        match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        if match:
            items.append({
                "rating": max(0, min(100, int(match.group(1)))),
                "flagged": match.group(2).upper() == "YES",
                "explanation": match.group(3).strip(),
            })
        else:
            # Fallback: isolate the block for item i and try loose extraction
            block_pattern = rf"(ITEM\s*{i}\b.+?)(?=ITEM\s*{i+1}\b|$)"
            block_match = re.search(block_pattern, raw, re.DOTALL | re.IGNORECASE)
            if block_match:
                fallback = _try_fallback_parse(block_match.group(1))
                items.append(fallback if fallback else {"rating": 50, "flagged": False, "explanation": ""})
            else:
                # No block found for this item — return safe default
                items.append({"rating": 50, "flagged": False, "explanation": ""})

    return items


def evaluate_phase2(
    parsed_ratings: list[dict],
    actual_correctness: list[bool],
    planted_mask: list[bool],
    planted_correct_mask: list[bool],
) -> Phase2Results:
    """Compute AUROC and planted detection rate for Phase 2.

    Args:
        parsed_ratings: List of dicts from parse_phase2_response.
        actual_correctness: Ground-truth boolean per item.
        planted_mask: Boolean mask — True for planted items.
        planted_correct_mask: Boolean mask — True for planted-correct items.

    Returns:
        Phase2Results dataclass with all audit metrics.
    """
    n = len(parsed_ratings)
    ratings_list = []
    scores, truths = [], []

    for i, (pr, correct) in enumerate(zip(parsed_ratings, actual_correctness)):
        ratings_list.append(AuditRating(f"item_{i}", pr["rating"], pr["flagged"], pr["explanation"]))
        scores.append(pr["rating"] / 100.0)
        truths.append(1 if correct else 0)

    try:
        auroc = roc_auc_score(truths, scores)
    except ValueError:
        auroc = 0.5

    planted_incorrect_idx = [i for i in range(n) if planted_mask[i] and not planted_correct_mask[i]]
    if planted_incorrect_idx:
        detected = sum(1 for i in planted_incorrect_idx if parsed_ratings[i]["flagged"])
        planted_detection = detected / len(planted_incorrect_idx)
    else:
        planted_detection = 0.0

    return Phase2Results(auroc, planted_detection, n, ratings_list, actual_correctness)
