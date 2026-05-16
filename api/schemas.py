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
    target_report_no: str
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
    excluded_ingredients: List[str] = Field(default_factory=list)
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


class UploadRecommendationResponse(BaseModel):
    input_type: str
    ocr: Optional[OCRPayload] = None
    parsed: ParsedOCRPayload
    detected_functional_ingredients: List[DetectedFunctionalIngredient] = Field(default_factory=list)
    estimated_profile: EstimatedProfilePayload
    recommendations: List[RecommendationItem] = Field(default_factory=list)
    execution_seconds: float = 0.0
    needs_user_review: bool = False
    review_message: str = ""
    quality_warnings: List[Dict[str, Any]] = Field(default_factory=list)
    parsed_ingredients_for_review: List[str] = Field(default_factory=list)
    excluded_ingredients: List[str] = Field(default_factory=list)
    product_name_category_hint: str = ""
