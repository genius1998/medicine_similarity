# API Usage

## Overview

현재 추천 엔진을 FastAPI로 감싼 최소 API 서버입니다. 기존 `product_function_profile.csv`, 제품 벡터 CSV, SQLite 캐시 DB를 읽고, 단일 제품 기준 TOP-K 추천과 explanation을 제공합니다.

## Install

```bash
pip install -r requirements_api.txt
```

## Runtime Files

- 제품 벡터 CSV: `c003_product_functional_vectors_final_rebuilt.csv`
- 제품 프로필 CSV: `output/product_function_profile.csv`
- SQLite DB: `ingredient_match_cache_rebuilt_item_class_i0050_final.sqlite`

경로는 기존 `config.json` 또는 스크립트 기본 경로 해석을 그대로 따릅니다.

## Run

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

- `GET /health`
- `GET /products`
- `GET /products/{report_no}`
- `GET /api/catalog/products?q=&page=1&page_size=20`
- `GET /api/catalog/products/{report_no}`
- `GET /api/products/search?q=검색어&limit=20`
- `GET /api/products/{report_no}/profile`
- `GET /api/products/{report_no}/similar?top_k=10&candidate_limit=1000&force_refresh=false`
- `POST /api/recommend/by-ingredients`

## Curl Examples

```bash
curl "http://localhost:8000/health"
curl "http://localhost:8000/api/catalog/products?page=1&page_size=20"
curl "http://localhost:8000/api/products/search?q=관절연골케어엔NEM"
curl "http://localhost:8000/api/products/2018001205671/profile"
curl "http://localhost:8000/api/products/2018001205671/similar?top_k=10&candidate_limit=1000"
```

```bash
curl -X POST "http://localhost:8000/api/recommend/by-ingredients" ^
  -H "Content-Type: application/json" ^
  -d "{\"raw_ingredients\":\"루테인, 지아잔틴, 비타민A, 아연\", \"top_k\":10, \"candidate_limit\":1000}"
```

## Cache Behavior

- 추천 API는 먼저 `product_similarity_explanation_cache`를 조회합니다.
- 캐시가 충분하면 그대로 반환합니다.
- 캐시가 없거나 `force_refresh=true`이면 단일 제품 추천을 다시 계산하고 SQLite를 갱신합니다.

## Duplicate Product Names

- 제품명 검색 API는 동일 제품명 다건을 모두 반환합니다.
- API는 검색 단계에서 임의로 하나를 확정하지 않습니다.
- 실제 추천은 사용자가 선택한 `report_no` 기준으로 호출해야 합니다.

## Catalog UI

- `/products` 에서 C003 전체 품목 목록을 페이지 단위로 조회할 수 있습니다.
- 각 행의 `상세 조회` 버튼은 `/products/{report_no}` 로 이동합니다.
- 상세 페이지에서는 C003 원문 상세 정보와 추천 엔진 결과를 함께 보여줍니다.

## Safety Notice

기본 caution 문구:

> 본 추천은 건강기능식품의 기능성원료, 표시 기능, 원재료 정보를 기준으로 한 비교 결과입니다. 질병의 예방·치료 효과를 의미하지 않으며, 실제 섭취 전 제품 라벨의 함량, 1일 섭취량, 주의사항을 확인해야 합니다.

추천 설명은 질병 치료나 의약품 대체 의미로 사용하지 않습니다.
