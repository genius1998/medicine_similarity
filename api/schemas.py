from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class ProductSearchItem(BaseModel):
    report_no: str
    product_name: str
    product_main_category: str
    primary_ingredients: List[str]
    secondary_ingredients: List[str]
    support_ingredients: List[str]
    confidence: Optional[float] = None
    notes: Optional[str] = None


class ProductSearchResponse(BaseModel):
    query: str
    count: int
    results: List[ProductSearchItem]


class ProductProfileResponse(BaseModel):
    report_no: str
    product_name: str
    product_main_category: str
    product_sub_categories: List[str]
    primary_ingredients: List[str]
    secondary_ingredients: List[str]
    support_ingredients: List[str]
    confidence: Optional[float] = None
    notes: Optional[str] = None


class RecommendationItem(BaseModel):
    rank: int
    report_no: str = ""
    target_report_no: str
    product_name: str = ""
    target_product_name: str
    similarity_score: float
    function_similarity_score: float
    core_match_score: float
    substitutability: str
    shared_ingredients: List[str]
    shared_categories: List[str]
    reason: str
    caution: str
    explanation: Dict[str, Any]
    exact_same_upload: bool = False
    uploaded_match_types: List[str] = Field(default_factory=list)
    uploaded_match_labels: List[str] = Field(default_factory=list)
    uploaded_status: str = ""
    uploaded_quality_grade: str = ""


class RecommendationBaseProduct(BaseModel):
    report_no: str
    product_name: str
    product_main_category: str
    primary_ingredients: List[str]


class RecommendationResponse(BaseModel):
    base_product: RecommendationBaseProduct
    recommendations: List[RecommendationItem]
    cache_used: bool
    execution_seconds: float


class IngredientRecommendationRequest(BaseModel):
    ingredients: List[str] = Field(default_factory=list)
    raw_ingredients: str = ""
    top_k: int = 10
    candidate_limit: int = 1000


class OCRTextRecommendationRequest(BaseModel):
    ocr_text: str = Field(..., min_length=1)
    top_k: int = 10
    candidate_limit: int = 1000


class ParsedOCRPayload(BaseModel):
    product_name_candidate: str = ""
    ingredient_section_text: str = ""
    functional_ingredient_candidates: List[str] = Field(default_factory=list)
    raw_ingredients: List[str] = Field(default_factory=list)
    normalized_ingredients: List[str] = Field(default_factory=list)
    ingredient_objects: List[Dict[str, Any]] = Field(default_factory=list)
    primary_ingredients: List[str] = Field(default_factory=list)
    primary_ingredients_normalized: List[str] = Field(default_factory=list)
    excluded_ingredients: List[str] = Field(default_factory=list)
    excluded_ingredient_objects: List[Dict[str, Any]] = Field(default_factory=list)
    quality_warnings: List[Dict[str, Any]] = Field(default_factory=list)
    nutrition_or_active_components: List[str] = Field(default_factory=list)
    daily_intake_text: str = ""
    warnings: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    needs_user_review: bool = False


class OCRPayload(BaseModel):
    raw_text: str = ""
    lines: List[str] = Field(default_factory=list)
    blocks: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: Optional[float] = None
    confidence_source: str = "unavailable"
    source: str = ""
    error: Optional[str] = None


class DetectedFunctionalIngredient(BaseModel):
    raw_ingredient: str
    functional_ingredient_name: str
    relation_type: str
    confidence: float
    display_name: str = ""
    normalized_for_matching: str = ""


class EstimatedProfilePayload(BaseModel):
    product_main_category: str = ""
    primary_ingredients: List[str] = Field(default_factory=list)
    secondary_ingredients: List[str] = Field(default_factory=list)
    support_ingredients: List[str] = Field(default_factory=list)
    upload_signature: str = ""
    profile_signature: str = ""
    status: str = ""
    quality_grade: str = ""
    is_candidate_enabled: Optional[bool] = None
    candidate_scope: str = ""
    candidate_disabled_reason: str = ""
    confidence: Optional[float] = None


class RecommendationQualityPayload(BaseModel):
    ocr_confidence: Optional[float] = None
    profile_confidence: float = 0.0
    quality_grade: str = ""
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    candidate_enabled: Optional[bool] = None
    status: str = ""
    candidate_scope: str = ""
    candidate_disabled_reason: str = ""


class RecommendationDebugPayload(BaseModel):
    image_hash: str = ""
    ocr_text_hash: str = ""
    parsed_signature: str = ""
    profile_signature: str = ""
    parse_cache_hit: bool = False
    exact_match_detected: bool = False


class SavedUploadedProduct(BaseModel):
    report_no: str
    product_name: str
    included_in_recommendations: bool = True


class UploadRecommendationResponse(BaseModel):
    input_type: str
    trace_id: str = ""
    ocr: Optional[OCRPayload] = None
    parsed: ParsedOCRPayload
    detected_functional_ingredients: List[DetectedFunctionalIngredient] = Field(default_factory=list)
    estimated_profile: EstimatedProfilePayload
    recommendations: List[RecommendationItem] = Field(default_factory=list)
    official_recommendations: List[RecommendationItem] = Field(default_factory=list)
    uploaded_similar_cases: List[RecommendationItem] = Field(default_factory=list)
    execution_seconds: float = 0.0
    needs_user_review: bool = False
    review_message: str = ""
    quality_warnings: List[Dict[str, Any]] = Field(default_factory=list)
    critical_warnings: List[Dict[str, Any]] = Field(default_factory=list)
    notices: List[Dict[str, Any]] = Field(default_factory=list)
    info_warnings: List[Dict[str, Any]] = Field(default_factory=list)
    parsed_ingredients_for_review: List[str] = Field(default_factory=list)
    excluded_ingredients: List[str] = Field(default_factory=list)
    product_name_category_hint: str = ""
    category_diversity_count: int = 0
    quality: RecommendationQualityPayload = Field(default_factory=RecommendationQualityPayload)
    debug: Optional[RecommendationDebugPayload] = None
    saved_product: Optional[SavedUploadedProduct] = None
