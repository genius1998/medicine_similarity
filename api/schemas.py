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
    raw_ingredients: str = Field(..., min_length=1)
    top_k: int = 10
    candidate_limit: int = 1000


class IngredientRecommendationResponse(BaseModel):
    input_ingredients: List[str]
    detected_functional_ingredients: List[str]
    estimated_main_category: Optional[str] = None
    recommendations: List[RecommendationItem] = Field(default_factory=list)
    not_implemented: bool = True
