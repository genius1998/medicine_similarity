from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List


def normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", "", text)


def search_profile_records(records: List[Dict], query: str, limit: int) -> List[Dict]:
    normalized_query = normalize_text(query)
    exact_matches = []
    contains_matches = []
    fuzzy_matches = []

    for record in records:
        product_name = str(record.get("product_name", "") or "")
        normalized_name = normalize_text(product_name)
        if not product_name:
            continue
        if product_name == query:
            exact_matches.append(record)
            continue
        if query in product_name or normalized_query in normalized_name:
            contains_matches.append(record)
            continue
        score = SequenceMatcher(None, normalized_query, normalized_name).ratio()
        if score >= 0.55:
            enriched = dict(record)
            enriched["_match_score"] = score
            fuzzy_matches.append(enriched)

    exact_matches = sorted(exact_matches, key=lambda item: (str(item.get("report_no", "")), str(item.get("product_name", ""))))
    contains_matches = sorted(
        contains_matches,
        key=lambda item: (
            abs(len(str(item.get("product_name", ""))) - len(query)),
            str(item.get("report_no", "")),
            str(item.get("product_name", "")),
        ),
    )
    fuzzy_matches = sorted(
        fuzzy_matches,
        key=lambda item: (-float(item.get("_match_score", 0.0)), str(item.get("report_no", "")), str(item.get("product_name", ""))),
    )
    combined = exact_matches + contains_matches + fuzzy_matches

    unique_rows = []
    seen = set()
    for item in combined:
        key = (str(item.get("report_no", "")), str(item.get("product_name", "")))
        if key in seen:
            continue
        seen.add(key)
        item = dict(item)
        item.pop("_match_score", None)
        unique_rows.append(item)
        if len(unique_rows) >= limit:
            break
    return unique_rows
