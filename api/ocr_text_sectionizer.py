from __future__ import annotations

import re
from typing import Dict, List, Tuple


SECTION_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("ingredient_area", ["원재료", "원료명", "원재료명 및 함량", "원료명 및 함량", "주원료"]),
    ("product_name_area", ["제품명", "상품명", "식품의 유형", "식품유형", "제품의 유형", "제품유형"]),
    ("functional_info_area", ["영양", "기능정보", "영양·기능정보", "기능성분", "기능성 원료"]),
    ("intake_area", ["섭취량", "섭취방법", "섭취 시", "섭취시", "1일 섭취량"]),
    ("warning_area", ["주의사항", "알레르기", "보관방법", "질병의 예방", "의약품이 아닙니다"]),
    ("company_area", ["제조원", "판매원", "유통전문판매원", "소재지"]),
]


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", str(line or "")).strip()


def _match_section(line: str) -> str:
    normalized = _normalize_line(line).lower()
    for section_name, keywords in SECTION_KEYWORDS:
        for keyword in keywords:
            if keyword.lower() in normalized:
                return section_name
    return ""


def sectionize_ocr_text(raw_text: str) -> Dict[str, object]:
    lines = [_normalize_line(line) for line in str(raw_text or "").splitlines()]
    lines = [line for line in lines if line]
    sections: Dict[str, List[str]] = {
        "product_name_area": [],
        "ingredient_area": [],
        "functional_info_area": [],
        "intake_area": [],
        "warning_area": [],
        "company_area": [],
        "unknown_area": [],
    }
    current_section = "product_name_area"
    for idx, line in enumerate(lines):
        matched_section = _match_section(line)
        if matched_section:
            current_section = matched_section
        if idx == 0 and not matched_section:
            sections["product_name_area"].append(line)
            continue
        sections[current_section].append(line)
    return {
        "sections": {name: "\n".join(values).strip() for name, values in sections.items()},
        "section_line_counts": {name: len(values) for name, values in sections.items()},
        "detected_section_names": [name for name, values in sections.items() if values],
        "raw_line_count": len(lines),
    }
