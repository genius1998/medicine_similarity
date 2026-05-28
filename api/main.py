from __future__ import annotations

import sqlite3
import time
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from api.config import get_settings
from api.ops_service import AuthUser, get_ops_service
from api.recommendation_service import RecommendationService
from api.schemas import (
    HealthResponse,
    IngredientRecommendationRequest,
    OCRTextRecommendationRequest,
    ProductProfileResponse,
    ProductSearchItem,
    ProductSearchResponse,
    RecommendationBaseProduct,
    RecommendationItem,
    RecommendationResponse,
    UploadRecommendationResponse,
)
from api.upload_recommendation_service import UploadRecommendationService, coerce_ingredient_request_payload


SESSION_COOKIE_NAME = "sb_session"

settings = get_settings()
app = FastAPI(title=settings.service_name)
service = RecommendationService()
upload_service = UploadRecommendationService(service)
ops_service = get_ops_service()
templates = Jinja2Templates(directory=str(settings.templates_dir))


def _user_to_dict(user: Optional[AuthUser]) -> Optional[dict]:
    if not user:
        return None
    return {
        "user_id": user.user_id,
        "email": user.email,
        "full_name": user.full_name,
        "organization": user.organization,
        "role": user.role,
        "is_active": user.is_active,
    }


def get_current_user(request: Request) -> Optional[AuthUser]:
    token = str(request.cookies.get(SESSION_COOKIE_NAME, "") or "").strip()
    if not token:
        return None
    return ops_service.get_user_by_session(token)


def template_context(request: Request, **extra) -> dict:
    user = get_current_user(request)
    context = {"current_user": _user_to_dict(user)}
    context.update(extra)
    return context


def require_page_user(request: Request) -> Optional[AuthUser]:
    return get_current_user(request)


def require_api_user(request: Request) -> AuthUser:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="login required")
    return user


def require_admin_user(request: Request) -> AuthUser:
    user = require_api_user(request)
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="admin required")
    return user


def login_redirect() -> RedirectResponse:
    return RedirectResponse(url="/login", status_code=303)


@app.middleware("http")
async def audit_request_middleware(request: Request, call_next):
    started = time.perf_counter()
    user = get_current_user(request)
    try:
        response = await call_next(request)
        ops_service.log_event(
            event_type="http_request",
            level="info",
            user_id=user.user_id if user else None,
            request_path=request.url.path,
            request_method=request.method,
            status_code=response.status_code,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
        return response
    except Exception as exc:  # noqa: BLE001
        ops_service.log_event(
            event_type="http_request",
            level="error",
            user_id=user.user_id if user else None,
            request_path=request.url.path,
            request_method=request.method,
            status_code=500,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            message=str(exc),
        )
        raise


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service.health()
    return HealthResponse(status="ok", service=settings.service_name)


@app.get("/", response_class=HTMLResponse)
def root_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "home.html", template_context(request))


@app.get("/home", response_class=HTMLResponse)
def home_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "home.html", template_context(request))


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: str = Query(""), joined: str = Query("")) -> HTMLResponse:
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/home", status_code=303)
    context = template_context(
        request,
        error_message=error,
        joined_message="회원가입이 완료되었습니다. 로그인해주세요." if joined else "",
        default_admin_email="admin@seongbun.local",
        default_admin_password="admin1234!",
    )
    return templates.TemplateResponse(request, "login.html", context)


@app.post("/login")
def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    user = ops_service.authenticate_user(email, password)
    if not user:
        ops_service.log_event(event_type="auth_login", level="warning", request_path="/login", request_method="POST", message=f"login failed for {email}")
        return RedirectResponse(url="/login?error=이메일 또는 비밀번호가 올바르지 않습니다.", status_code=303)
    token = ops_service.create_session(user.user_id)
    ops_service.log_event(event_type="auth_login", level="info", user_id=user.user_id, request_path="/login", request_method="POST", message="login success")
    response = RedirectResponse(url="/home", status_code=303)
    response.set_cookie(SESSION_COOKIE_NAME, token, httponly=True, samesite="lax")
    return response


@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request, error: str = Query("")) -> HTMLResponse:
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/home", status_code=303)
    return templates.TemplateResponse(request, "signup.html", template_context(request, error_message=error))


@app.post("/signup")
def signup_submit(
    request: Request,
    full_name: str = Form(""),
    organization: str = Form(""),
    email: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
):
    if password != password_confirm:
        return RedirectResponse(url="/signup?error=비밀번호 확인이 일치하지 않습니다.", status_code=303)
    try:
        user = ops_service.create_user(email=email, password=password, full_name=full_name, organization=organization, role="user")
        ops_service.log_event(event_type="auth_signup", level="info", user_id=user.user_id, request_path="/signup", request_method="POST", message="signup success")
    except sqlite3.IntegrityError:
        return RedirectResponse(url="/signup?error=이미 사용 중인 이메일입니다.", status_code=303)
    except ValueError as exc:
        return RedirectResponse(url=f"/signup?error={str(exc)}", status_code=303)
    return RedirectResponse(url="/login?joined=1", status_code=303)


@app.post("/logout")
def logout_submit(request: Request):
    token = str(request.cookies.get(SESSION_COOKIE_NAME, "") or "")
    user = get_current_user(request)
    if token:
        ops_service.delete_session(token)
    ops_service.log_event(event_type="auth_logout", level="info", user_id=user.user_id if user else None, request_path="/logout", request_method="POST", message="logout")
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response


@app.get("/products", response_class=HTMLResponse)
def products_page(request: Request) -> HTMLResponse:
    user = require_page_user(request)
    if not user:
        return login_redirect()
    return templates.TemplateResponse(request, "products.html", template_context(request))


@app.get("/products/{report_no}", response_class=HTMLResponse)
def product_detail_page(request: Request, report_no: str) -> HTMLResponse:
    user = require_page_user(request)
    if not user:
        return login_redirect()
    return templates.TemplateResponse(request, "product_detail.html", template_context(request, report_no=report_no))


@app.get("/recommend/image", response_class=HTMLResponse)
def image_recommend_page(request: Request) -> HTMLResponse:
    user = require_page_user(request)
    if not user:
        return login_redirect()
    return templates.TemplateResponse(
        request,
        "image_recommend.html",
        template_context(
            request,
            api_base_url="",
            default_top_k=settings.default_top_k,
            default_candidate_limit=settings.default_candidate_limit,
            debug_response_enabled=settings.debug_response,
        ),
    )


@app.get("/ocr-recommend", response_class=HTMLResponse)
def ocr_recommend_page(request: Request) -> HTMLResponse:
    return image_recommend_page(request)


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request) -> HTMLResponse:
    user = get_current_user(request)
    if not user:
        return login_redirect()
    if user.role != "admin":
        return RedirectResponse(url="/home", status_code=303)
    context = template_context(
        request,
        monitoring=ops_service.get_monitoring_snapshot(),
        pending_ingredients=ops_service.list_pending_ingredient_approvals(limit=200, status="pending"),
        ocr_reviews=ops_service.list_ocr_reviews(limit=200),
        app_logs=ops_service.list_app_logs(limit=200),
    )
    return templates.TemplateResponse(request, "admin.html", context)


@app.post("/admin/ingredients/review")
def review_pending_ingredient(
    request: Request,
    functional_ingredient_name: str = Form(...),
    decision: str = Form(...),
    category_main: str = Form(""),
    category_sub: str = Form(""),
    claim_text: str = Form(""),
    notes: str = Form(""),
):
    user = require_admin_user(request)
    ops_service.review_pending_ingredient(
        functional_ingredient_name=functional_ingredient_name,
        decision=decision,
        reviewer_user_id=user.user_id,
        category_main=category_main,
        category_sub=category_sub,
        claim_text=claim_text,
        notes=notes,
    )
    ops_service.log_event(
        event_type="ingredient_review",
        level="info",
        user_id=user.user_id,
        request_path="/admin/ingredients/review",
        request_method="POST",
        message=f"{decision}:{functional_ingredient_name}",
    )
    return RedirectResponse(url="/admin", status_code=303)


@app.get("/admin/export/ocr-reviews.xlsx")
def export_ocr_reviews(request: Request):
    require_admin_user(request)
    payload = ops_service.export_ocr_reviews_xlsx()
    return StreamingResponse(
        BytesIO(payload),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="ocr_reviews.xlsx"'},
    )


@app.get("/admin/export/pending-ingredients.xlsx")
def export_pending_ingredients(request: Request):
    require_admin_user(request)
    payload = ops_service.export_pending_ingredients_xlsx()
    return StreamingResponse(
        BytesIO(payload),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="pending_ingredients.xlsx"'},
    )


@app.get("/admin/export/app-logs.xlsx")
def export_app_logs(request: Request):
    require_admin_user(request)
    payload = ops_service.export_app_logs_xlsx()
    return StreamingResponse(
        BytesIO(payload),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="app_logs.xlsx"'},
    )


@app.get("/api/catalog/products")
def list_catalog_products(
    request: Request,
    q: str = Query("", min_length=0),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> dict:
    require_api_user(request)
    return service.list_catalog_products(q, page, page_size)


@app.get("/api/catalog/products/{report_no}")
def get_catalog_product_detail(request: Request, report_no: str) -> dict:
    require_api_user(request)
    detail = service.get_catalog_product_detail(report_no)
    if not detail:
        raise HTTPException(status_code=404, detail=f"report_no not found: {report_no}")
    return detail


@app.get("/api/products/search", response_model=ProductSearchResponse)
def search_products(request: Request, q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)) -> ProductSearchResponse:
    require_api_user(request)
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
def get_product_profile(request: Request, report_no: str) -> ProductProfileResponse:
    require_api_user(request)
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
    request: Request,
    report_no: str,
    top_k: int = Query(settings.default_top_k, ge=1, le=50),
    candidate_limit: int = Query(settings.default_candidate_limit, ge=10, le=5000),
    force_refresh: bool = Query(False),
    llm_rerank: bool = Query(False),
) -> RecommendationResponse:
    require_api_user(request)
    try:
        payload = service.get_similar_products(report_no, top_k, candidate_limit, force_refresh, llm_rerank)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RecommendationResponse(
        base_product=RecommendationBaseProduct(**payload["base_product"]),
        recommendations=[RecommendationItem(**item) for item in payload["recommendations"]],
        cache_used=bool(payload["cache_used"]),
        llm_rerank_applied=bool(payload.get("llm_rerank_applied", False)),
        llm_rerank_error=str(payload.get("llm_rerank_error", "") or ""),
        execution_seconds=float(payload["execution_seconds"]),
    )


@app.post("/api/recommend/by-ingredients", response_model=UploadRecommendationResponse)
def recommend_by_ingredients(request: Request, payload_in: IngredientRecommendationRequest) -> UploadRecommendationResponse:
    user = require_api_user(request)
    ingredients = coerce_ingredient_request_payload(payload_in.ingredients, payload_in.raw_ingredients)
    payload = upload_service.recommend_from_ingredients(
        ingredients,
        payload_in.top_k,
        payload_in.candidate_limit,
        product_name_candidate=payload_in.product_name_candidate,
        llm_rerank=payload_in.llm_rerank,
    )
    ops_service.save_ocr_review(
        user_id=user.user_id,
        source_type="ingredients",
        response_payload=payload,
        edited_ingredients=ingredients,
        base_trace_id=payload_in.base_trace_id,
        review_notes=payload_in.review_notes,
    )
    ops_service.log_event(
        event_type="ocr_recommend",
        level="info",
        user_id=user.user_id,
        request_path="/api/recommend/by-ingredients",
        request_method="POST",
        message="edited ingredient rerank",
        payload={"ingredient_count": len(ingredients), "trace_id": payload.get("trace_id", "")},
    )
    return UploadRecommendationResponse(**payload)


@app.post("/api/recommend/by-ocr-text", response_model=UploadRecommendationResponse)
def recommend_by_ocr_text(request: Request, payload_in: OCRTextRecommendationRequest) -> UploadRecommendationResponse:
    user = require_api_user(request)
    payload = upload_service.recommend_from_ocr_text(payload_in.ocr_text, payload_in.top_k, payload_in.candidate_limit, llm_rerank=payload_in.llm_rerank)
    ops_service.save_ocr_review(user_id=user.user_id, source_type="ocr_text", response_payload=payload)
    ops_service.log_event(
        event_type="ocr_recommend",
        level="info",
        user_id=user.user_id,
        request_path="/api/recommend/by-ocr-text",
        request_method="POST",
        message="ocr text recommend",
        payload={"trace_id": payload.get("trace_id", "")},
    )
    return UploadRecommendationResponse(**payload)


@app.post("/api/recommend/by-image", response_model=UploadRecommendationResponse)
async def recommend_by_image(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Form(settings.default_top_k),
    candidate_limit: int = Form(settings.default_candidate_limit),
    llm_rerank: bool = Form(False),
) -> UploadRecommendationResponse:
    user = require_api_user(request)
    filename = str(file.filename or "upload.jpg")
    suffix = filename[filename.rfind(".") :].lower() if "." in filename else ""
    if suffix not in settings.upload_allowed_extensions:
        raise HTTPException(status_code=400, detail=f"unsupported file extension: {suffix}")

    image_bytes = await file.read()
    max_size = settings.upload_max_file_size_mb * 1024 * 1024
    if len(image_bytes) > max_size:
        raise HTTPException(status_code=400, detail=f"file too large: max {settings.upload_max_file_size_mb}MB")

    payload = upload_service.recommend_from_uploaded_image(image_bytes, filename, top_k, candidate_limit, llm_rerank=llm_rerank)
    ops_service.save_ocr_review(user_id=user.user_id, source_type="image", response_payload=payload)
    ops_service.log_event(
        event_type="ocr_recommend",
        level="info",
        user_id=user.user_id,
        request_path="/api/recommend/by-image",
        request_method="POST",
        message=f"image recommend:{filename}",
        payload={"trace_id": payload.get("trace_id", ""), "filename": filename},
    )
    return UploadRecommendationResponse(**payload)
