from __future__ import annotations

import re
from typing import Iterable


def normalize_family_key(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


JOINT_INGREDIENT_FAMILY_RULES = {
    "chondroitin": {
        "tokens": (
            "콘드로이친황산염",
            "콘드로이친황산",
            "콘드로이친",
            "chondroitinsulfate",
            "chondroitin",
        ),
        "canonical_name": "콘드로이친 황산염",
        "aliases": (
            "콘드로이친 황산염",
            "콘드로이친 황산염(제2020-1호)",
            "콘드로이친황산염",
            "콘드로이친",
        ),
    },
    "glucosamine": {
        "tokens": (
            "글루코사민",
            "glucosamine",
        ),
        "canonical_name": "글루코사민",
        "aliases": (
            "글루코사민",
            "글루코사민염산염",
            "글루코사민 황산염",
        ),
    },
    "nag": {
        "tokens": (
            "n-acetylglucosamine",
            "nacetylglucosamine",
            "n-아세틸글루코사민",
            "n아세틸글루코사민",
            "엔에이지",
            "nag",
        ),
        "canonical_name": "NAG(엔에이지, N-아세틸글루코사민, N-Acetylglucosamine)",
        "aliases": (
            "NAG",
            "NAG(엔에이지, N-아세틸글루코사민, N-Acetylglucosamine)",
            "N-아세틸글루코사민",
            "엔에이지",
        ),
    },
    "uc_ii": {
        "tokens": (
            "uc-ii",
            "ucii",
            "닭가슴연골분말",
            "닭가슴연골",
            "닭연골",
            "닭벼슬",
            "chickensternumcartilage",
            "undenaturedtypeiicollagen",
        ),
        "canonical_name": "닭가슴연골분말(UC-II)",
        "aliases": (
            "닭가슴연골분말(UC-II)",
            "닭가슴연골분말(UC-II)(제2014-39호)",
            "닭가슴연골분말(UC-II)(기능성원료인정제2014-39호)",
            "UC-II",
            "닭가슴연골분말",
            "닭연골",
        ),
    },
    "msm": {
        "tokens": (
            "msm",
            "엠에스엠",
            "dimethylsulfone",
            "methylsulfonylmethane",
        ),
        "canonical_name": "MSM",
        "aliases": (
            "MSM",
            "엠에스엠(MSM, Methyl sulfonylmethane, 디메틸설폰)",
            "Dimethylsulfone (MSM)",
            "Dimethylsulfone(MSM)",
        ),
    },
    "boswellia": {
        "tokens": (
            "보스웰리아",
            "boswellia",
        ),
        "canonical_name": "보스웰리아추출물",
        "aliases": (
            "보스웰리아추출물",
            "보스웰리아",
        ),
    },
    "shark_cartilage": {
        "tokens": (
            "\uc0c1\uc5b4\uc5f0\uace8",
            "\ubb34\ucf54\ub2e4\ub2f9",
            "\ubba4\ucf54\ub2e4\ub2f9",
            "mucopolysaccharide",
            "sharkcartilage",
        ),
        "canonical_name": "\ubba4\ucf54\ub2e4\ub2f9.\ub2e8\ubc31",
        "aliases": (
            "\ubba4\ucf54\ub2e4\ub2f9.\ub2e8\ubc31",
            "\uc0c1\uc5b4\uc5f0\uace8",
            "\uc0c1\uc5b4\uc5f0\uace8\ubd84\ub9d0",
            "\uc0c1\uc5b4\uc5f0\uace8\ucd94\ucd9c\ubb3c\ubd84\ub9d0",
        ),
    },
    "achyranthes": {
        "tokens": (
            "\uc1e0\ubb34\ub98e",
            "\uc1e0\ubb34\ub985",
            "\uc6b0\uc2ac",
            "hl-joint100",
            "hljoint100",
            "achyranthes",
        ),
        "canonical_name": "\uc6b0\uc2ac \ub4f1 \ubcf5\ud569\ubb3c(HL-Joint100)",
        "aliases": (
            "\uc6b0\uc2ac \ub4f1 \ubcf5\ud569\ubb3c(HL-Joint100)",
            "\uc1e0\ubb34\ub98e\ubd84\ub9d0",
            "\uc1e0\ubb34\ub985\ubd84\ub9d0",
            "\uc6b0\uc2ac",
        ),
    },
}

PROTECTED_JOINT_FAMILIES = frozenset(JOINT_INGREDIENT_FAMILY_RULES)


def infer_joint_ingredient_family(value: str) -> str:
    normalized = normalize_family_key(value)
    if not normalized:
        return ""
    for family, rule in JOINT_INGREDIENT_FAMILY_RULES.items():
        if any(token in normalized for token in rule["tokens"]):
            return family
    return ""


def canonical_joint_family_name(family: str) -> str:
    rule = JOINT_INGREDIENT_FAMILY_RULES.get(str(family or ""))
    return str(rule["canonical_name"]) if rule else ""


def joint_family_aliases(family: str) -> tuple[str, ...]:
    rule = JOINT_INGREDIENT_FAMILY_RULES.get(str(family or ""))
    if not rule:
        return ()
    return tuple(str(value) for value in rule["aliases"])


def collect_joint_ingredient_families(values: Iterable[str]) -> set[str]:
    families = set()
    for value in values:
        family = infer_joint_ingredient_family(str(value or ""))
        if family:
            families.add(family)
    return families
