from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api.config import get_settings
from api.recommendation_service import RecommendationService
from api.schemas import (
    HealthResponse,
    IngredientRecommendationRequest,
    IngredientRecommendationResponse,
    ProductProfileResponse,
    ProductSearchItem,
    ProductSearchResponse,
    RecommendationBaseProduct,
    RecommendationItem,
    RecommendationResponse,
)


settings = get_settings()
app = FastAPI(title=settings.service_name)
service = RecommendationService()
templates = Jinja2Templates(directory=str(settings.templates_dir))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service.health()
    return HealthResponse(status="ok", service=settings.service_name)


@app.get("/", response_class=HTMLResponse)
def root_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("products.html", {"request": request})


@app.get("/products", response_class=HTMLResponse)
def products_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("products.html", {"request": request})


@app.get("/products/{report_no}", response_class=HTMLResponse)
def product_detail_page(request: Request, report_no: str) -> HTMLResponse:
    return templates.TemplateResponse("product_detail.html", {"request": request, "report_no": report_no})


@app.get("/api/catalog/products")
def list_catalog_products(
    q: str = Query("", min_length=0),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict:
    return service.list_catalog_products(q, page, page_size)


@app.get("/api/catalog/products/{report_no}")
def get_catalog_product_detail(report_no: str) -> dict:
    detail = service.get_catalog_product_detail(report_no)
    if not detail:
        raise HTTPException(status_code=404, detail=f"report_no not found: {report_no}")
    return detail


@app.get("/api/products/search", response_model=ProductSearchResponse)
def search_products(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)) -> ProductSearchResponse:
    results = service.search_products(q, limit)
    items = [
        ProductSearchItem(
            report_no=str(item.get("report_no", "")),
            product_name=str(item.get("product_name", "")),
            product_main_category=str(item.get("product_main_category", "")),
            primary_ingredients=list(item.get("primary_ingredients", [])),
            secondary_ingredients=list(item.get("secondary_ingredients", [])),
            support_ingredients=list(item.get("support_ingredients", [])),
            confidence=float(item.get("confidence", 0.0) or 0.0),
            notes=str(item.get("notes", "") or ""),
        )
        for item in results
    ]
    return ProductSearchResponse(query=q, count=len(items), results=items)


@app.get("/api/products/{report_no}/profile", response_model=ProductProfileResponse)
def get_product_profile(report_no: str) -> ProductProfileResponse:
    profile = service.get_profile_by_report_no(report_no)
    if not profile:
        raise HTTPException(status_code=404, detail=f"report_no not found: {report_no}")
    return ProductProfileResponse(
        report_no=str(profile.get("report_no", "")),
        product_name=str(profile.get("product_name", "")),
        product_main_category=str(profile.get("product_main_category", "")),
        product_sub_categories=list(profile.get("product_sub_categories", [])),
        primary_ingredients=list(profile.get("primary_ingredients", [])),
        secondary_ingredients=list(profile.get("secondary_ingredients", [])),
        support_ingredients=list(profile.get("support_ingredients", [])),
        confidence=float(profile.get("confidence", 0.0) or 0.0),
        notes=str(profile.get("notes", "") or ""),
    )


@app.get("/api/products/{report_no}/similar", response_model=RecommendationResponse)
def get_similar_products(
    report_no: str,
    top_k: int = Query(settings.default_top_k, ge=1, le=50),
    candidate_limit: int = Query(settings.default_candidate_limit, ge=10, le=5000),
    force_refresh: bool = Query(False),
) -> RecommendationResponse:
    try:
        payload = service.get_similar_products(report_no, top_k, candidate_limit, force_refresh)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return RecommendationResponse(
        base_product=RecommendationBaseProduct(**payload["base_product"]),
        recommendations=[RecommendationItem(**item) for item in payload["recommendations"]],
        cache_used=bool(payload["cache_used"]),
        execution_seconds=float(payload["execution_seconds"]),
    )


@app.post("/api/recommend/by-ingredients", response_model=IngredientRecommendationResponse)
def recommend_by_ingredients(request: IngredientRecommendationRequest) -> IngredientRecommendationResponse:
    payload = service.recommend_by_ingredients(request.raw_ingredients, request.top_k, request.candidate_limit)
    return IngredientRecommendationResponse(**payload)
