"""
Ingredient matching constants.

All domain constants used by UploadRecommendationService and related modules
are centralised here so they can be imported without pulling in the full
service class.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Category hint rules (product-name / ingredient-name heuristics)
# ---------------------------------------------------------------------------

CATEGORY_HINT_RULES: dict[str, dict] = {
    "면역": {
        "product_keywords": ["홍삼", "고려홍삼", "6년근", "홍삼정", "진세노사이드", "홍삼스틱", "정관장"],
        "ingredient_keywords": ["홍삼", "홍삼농축액", "진세노사이드", "홍삼근", "홍삼분말"],
    },
    "눈 건강": {
        "product_keywords": ["루테인", "지아잔틴", "눈", "아이", "마리골드", "빌베리", "아스타잔틴", "베타카로틴"],
        "ingredient_keywords": ["루테인", "지아잔틴", "마리골드꽃추출물", "베타카로틴", "비타민 A", "빌베리추출물", "블루베리", "헤마토코쿠스추출물"],
    },
    "관절/연골": {
        "product_keywords": ["관절", "연골", "콘드로이친", "뮤코다당", "뮤코다당단백", "철갑상어", "철갑상어연골", "MSM", "보스웰리아", "NEM", "난각막", "강황"],
        "ingredient_keywords": ["콘드로이친", "뮤코다당단백", "철갑상어연골", "MSM", "엠에스엠", "보스웰리아", "난각막", "난각막분말", "난각막가수분해물", "강황추출물", "초록입홍합"],
    },
    "장 건강": {
        "product_keywords": ["유산균", "프로바이오틱스", "락토핏", "장", "대장", "프리바이오틱스", "올리고당"],
        "ingredient_keywords": ["프로바이오틱스", "유산균", "갈락토올리고당", "프락토올리고당", "난소화성말토덱스트린", "이눌린"],
    },
    "혈당": {
        "product_keywords": ["혈당", "당당자유", "당케어", "고투플러스"],
        "ingredient_keywords": ["바나바", "바나바잎", "바나바잎추출물", "코로솔산", "난소화성말토덱스트린", "구아바잎추출물", "달맞이꽃종자추출물", "구아검", "이눌린", "식이섬유"],
    },
    "체지방": {
        "product_keywords": ["체지방", "다이어트", "슬림", "컷", "컷팅"],
        "ingredient_keywords": ["키토산", "키토올리고당", "가르시니아", "hca", "공액리놀레산", "cla", "녹차추출물", "카테킨", "콜레우스포스콜리", "잔티젠"],
    },
    "피부 건강": {
        "product_keywords": ["콜라겐", "피부", "레티놀", "히알루론", "세라마이드", "비오틴"],
        "ingredient_keywords": ["콜라겐", "저분자콜라겐펩타이드", "콜라겐펩타이드", "비오틴", "비타민 A", "히알루론산", "세라마이드"],
    },
    "기억력/인지력": {
        "product_keywords": ["인지", "기억", "포스파티딜세린", "브레인", "두뇌"],
        "ingredient_keywords": ["포스파티딜세린", "은행잎", "테아닌", "DHA"],
    },
    "뼈 건강": {
        "product_keywords": ["뼈", "칼슘", "칼마디", "비타민D", "K2"],
        "ingredient_keywords": ["칼슘", "비타민 D", "비타민 K", "마그네슘"],
    },
    "남성 건강": {
        "product_keywords": ["전립선", "남성", "쏘팔메토", "옥타코사놀", "활력"],
        "ingredient_keywords": ["쏘팔메토", "쏘팔메토열매추출물", "로르산", "옥타코사놀", "아연", "녹용", "마카", "아르기닌"],
    },
    "간 건강": {
        "product_keywords": ["밀크씨슬", "밀크시슬", "실리마린", "간", "liver", "milk thistle"],
        "ingredient_keywords": ["밀크씨슬", "밀크시슬", "밀크씨슬추출물", "실리마린", "silymarin", "milk thistle"],
    },
}

# ---------------------------------------------------------------------------
# Primary-ingredient exclusion lists
# ---------------------------------------------------------------------------

EXCLUDED_PRIMARY_KEYWORDS: list[str] = [
    "히드록시프로필메틸셀룰로오스",
    "hpmc",
    "결정셀룰로오스",
    "미결정셀룰로오스",
    "스테아린산마그네슘",
    "이산화규소",
    "카복시메틸셀룰로오스",
    "글리세린지방산에스테르",
    "캡슐기제",
    "젤라틴",
    "글리세린",
    "착색료",
    "이산화티타늄",
    "말토덱스트린",
    "덱스트린",
]

NON_EXCLUDED_PRIMARY_KEYWORDS: list[str] = [
    "난소화성말토덱스트린",
    "난각막",
    "난각막분말",
    "난각막가수분해물",
    "nem",
    "demr",
    "난각칼슘",
    "난각분말",
    "계란껍질분말",
    "프로바이오틱스",
    "프리바이오틱스",
    "갈락토올리고당",
    "프락토올리고당",
    "이눌린",
]

# ---------------------------------------------------------------------------
# Role weights
# ---------------------------------------------------------------------------

ROLE_WEIGHT: dict[str, float] = {
    "primary": 1.0,
    "secondary": 0.72,
    "support": 0.45,
}

# ---------------------------------------------------------------------------
# Cache relation-type guards
# ---------------------------------------------------------------------------

# Relation types that should NEVER be used as positive ingredient matches.
# Entries with these types are rejected in _resolve_cache_row_match and fall
# through to RAG re-matching.
NON_FUNCTIONAL_CACHE_RELATION_TYPES: set[str] = {
    "excipient",
    "unrelated",
    "unknown",
    "error",
    "same_function_only",  # same functional category but DIFFERENT ingredient
}

# Relation types permitted in the upload similarity vector.
VECTOR_ALLOWED_RELATION_TYPES: set[str] = {
    "same_ingredient",
    "marker_compound",
    "ingredient_group",
    "nutrient_form",
}

# ---------------------------------------------------------------------------
# OCR / parse quality thresholds
# ---------------------------------------------------------------------------

LOW_OCR_CONFIDENCE_THRESHOLD: float = 0.6
LOW_PARSE_CONFIDENCE_THRESHOLD: float = 0.65
LOW_TOP_SIMILARITY_THRESHOLD: float = 0.25
MIN_SUFFICIENT_OCR_TEXT_LENGTH: int = 80
TOO_MANY_INGREDIENTS_THRESHOLD: int = 20

# ---------------------------------------------------------------------------
# Joint / supplement product & ingredient signal keywords
# ---------------------------------------------------------------------------

STRONG_JOINT_PRODUCT_KEYWORDS: list[str] = [
    "관절", "연골", "콘드로이친", "뮤코다당", "뮤코다당단백",
    "철갑상어", "철갑상어연골", "MSM", "보스웰리아", "NEM", "난각막",
]

STRONG_JOINT_INGREDIENT_KEYWORDS: list[str] = [
    "콘드로이친", "뮤코다당단백", "철갑상어연골", "MSM", "엠에스엠",
    "글루코사민", "NAG", "보스웰리아", "난각막", "난각막분말",
    "난각막가수분해물", "초록입홍합", "UC-II", "비변성콜라겐",
]

NUTRITION_SUPPLEMENT_PRODUCT_KEYWORDS: list[str] = [
    "멀티", "멀티밸런스", "멀티비타민", "비타민", "미네랄", "요오드",
]

NUTRITION_SUPPLEMENT_INGREDIENT_KEYWORDS: list[str] = [
    "비타민 A", "비타민 B1", "비타민 B2", "비타민 B6", "비타민 B12",
    "비타민 C", "비타민 D", "비타민 E", "비타민 K", "엽산", "나이아신",
    "비오틴", "판토텐산", "셀레늄", "아연", "칼슘", "마그네슘", "철분",
    "요오드", "망간", "크롬", "구리", "몰리브덴", "dl-a-토코페롤",
]

# ---------------------------------------------------------------------------
# Warning severity codes
# ---------------------------------------------------------------------------

CRITICAL_WARNING_CODES: set[str] = {
    "critical_warning",
    "ingredient_section_unclear",
    "ocr_error",
    "category_conflict_product_name_vs_ingredients",
    "generic_category_with_named_product",
    "low_parse_confidence",
    "low_ocr_confidence",
    "low_ocr_text_quality",
    "excipient_in_primary",
    "mixed_primary_categories",
    "low_top_similarity_score",
    "missing_recommendations",
    "top_reason_category_mismatch",
    "too_many_normalized_ingredients",
}

NOTICE_WARNING_CODES: set[str] = {
    "notice_warning",
    "allergen_notice",
}

INFO_WARNING_CODES: set[str] = {
    "info_warning",
    "ocr_confidence_unavailable",
}

# ---------------------------------------------------------------------------
# Runtime RAG / cache settings
# ---------------------------------------------------------------------------

ENABLE_RUNTIME_RAG_FALLBACK: bool = True
RUNTIME_CUSTOM_FUNCTIONAL_TABLE_NAME: str = "runtime_custom_functional_ingredient_map"
RUNTIME_RAG_TOP_K: int = 5
RUNTIME_RAG_NAME_SIMILARITY_MIN: float = 0.78

# Short / ambiguous keys that must never use LIKE cache lookups.
RUNTIME_LIKE_DISABLED_KEYS: set[str] = {
    "hca", "gaba", "cla", "msm", "epa", "dha", "cpp", "bcaa",
    "coq10", "mk7", "rg1", "rb1", "rg3", "l테아닌", "테아닌",
}

# ---------------------------------------------------------------------------
# Vector / excipient exclusion term sets
# ---------------------------------------------------------------------------

RUNTIME_CACHE_LIKE_VECTOR_EXCLUDED_TERMS: set[str] = {
    "식물혼합농축액", "식물혼합추출물", "혼합농축액", "종자추출물",
    "구연산", "구연산삼나트륨", "구연산나트륨", "전분", "옥수수",
    "말토덱스트린", "덱스트린", "정제수", "주정", "에탄올",
    "향료", "감미료", "혼합제제",
    "200mg", "500mg", "mg", "g", "%",
    "hpmc", "히드록시프로필메틸셀룰로오스", "카복시메틸셀룰로오스",
    "스테아린산마그네슘", "이산화규소", "결정셀룰로오스",
    "글리세린", "d소비탈액", "소비탈액",
}

RUNTIME_EXCIPIENT_VECTOR_EXCLUDED_TERMS: set[str] = {
    "히드록시프로필메틸셀룰로오스오스", "히드록시프로필메틸셀룰로오스",
    "hpmc", "결정셀룰로오스", "미결정셀룰로오스", "이산화규소",
    "스테아린산마그네슘", "스테아린산칼슘", "카복시메틸셀룰로오스",
    "카복시메틸셀룰로오스칼슘", "글리세린", "d소비탈액", "소비탈액",
    "덱스트린", "말토덱스트린", "정제수", "아라비아검", "설탕",
    "이소말트", "에리스리톨", "자일리톨", "향료", "착향료", "감미료",
}

RUNTIME_RAG_FALLBACK_SKIP_TERMS: set[str] = RUNTIME_EXCIPIENT_VECTOR_EXCLUDED_TERMS | {
    "가공유지", "유당", "유당혼합분말", "건조효모", "채소혼합분말",
    "채소호합분말", "야채혼합분말", "전분",
}

RUNTIME_LOW_SIGNAL_UPLOAD_VECTOR_EXACT_TERMS: set[str] = {
    "가공유지", "유당", "유당혼합분말", "건조효모",
    "채소혼합분말", "채소호합분말", "야채혼합분말", "전분",
}

RUNTIME_VECTOR_EXCLUSION_REASON_MESSAGES: dict[str, str] = {
    "excluded_from_vector_by_cache_like_guard": "cache_like guard excluded this match from the upload vector",
    "excluded_from_vector_by_excipient_guard": "excipient guard excluded this match from the upload vector",
    "excluded_from_vector_by_low_signal_upload_guard": "low-signal upload ingredient excluded from the upload vector",
    "excluded_from_vector_by_relation_guard": "relation type guard excluded this match from the upload vector",
}

# ---------------------------------------------------------------------------
# Suspicious-mapping guard rules
# ---------------------------------------------------------------------------

RUNTIME_SUSPICIOUS_MAPPING_RULES: list[dict] = [
    {
        "name": "hca",
        "inputs": ["hca", "hydroxycitricacid", "hydroxycitric"],
        "allowed": ["가르시니아", "garcinia", "hca"],
        "blocked": ["녹차", "카테킨", "egcg", "greentea"],
        "safe_aliases": [
            "가르시니아 캄보지아 껍질추출물 60HCA",
            "가르시니아 캄보지아 추출물",
            "가르시니아 캄보지아 껍질추출물",
        ],
    },
    {
        "name": "silymarin",
        "inputs": ["실리마린", "silymarin"],
        "allowed": ["밀크씨슬", "밀크시슬", "milkthistle", "silymarin"],
        "blocked": [],
        "safe_aliases": ["밀크씨슬추출물", "밀크씨슬 추출물", "밀크씨슬", "milk thistle"],
    },
    {
        "name": "ginsenoside",
        "inputs": ["진세노사이드", "ginsenoside"],
        "allowed": ["홍삼", "인삼", "ginseng", "redginseng"],
        "blocked": [],
        "safe_aliases": ["홍삼", "인삼"],
    },
    {
        "name": "lutein_zeaxanthin",
        "inputs": ["루테인", "lutein", "지아잔틴", "zeaxanthin"],
        "allowed": ["루테인", "지아잔틴", "마리골드", "lutein", "zeaxanthin", "marigold"],
        "blocked": [],
        "safe_aliases": ["루테인", "지아잔틴", "마리골드꽃추출물", "마리골드꽃추출물(지아잔틴함유)"],
    },
    {
        "name": "omega3",
        "inputs": ["epa", "dha", "오메가3", "오메가-3", "omega3", "fishoil", "fish oil"],
        "allowed": ["epa", "dha", "오메가", "omega", "어유", "fishoil"],
        "blocked": [],
        "safe_aliases": ["EPA 및 DHA 함유 유지", "EPA 및 DHA 함유 유지(고시형 원료)", "오메가-3 지방산 함유유지"],
    },
    {
        "name": "vitamin_d",
        "inputs": ["비타민d3", "건조비타민d3", "vitamind3", "vitamind"],
        "allowed": ["비타민d", "vitamind"],
        "blocked": [],
        "safe_aliases": ["비타민 D", "비타민D"],
    },
    {
        "name": "zinc",
        "inputs": ["산화아연", "글루콘산아연", "zincoxide", "zincgluconate"],
        "allowed": ["아연", "zinc"],
        "blocked": [],
        "safe_aliases": ["아연", "zinc"],
    },
]

# ---------------------------------------------------------------------------
# Joint ingredient family sub-category labels
# ---------------------------------------------------------------------------

JOINT_FAMILY_CATEGORY_SUB: dict[str, str] = {
    "chondroitin": "콘드로이친 계열",
    "glucosamine": "글루코사민 계열",
    "nag": "NAG 계열",
    "uc_ii": "UC-II 계열",
    "msm": "MSM 계열",
    "boswellia": "보스웰리아 계열",
}
