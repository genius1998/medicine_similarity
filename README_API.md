# API Usage

## Overview

이 API 서버는 건강기능식품 추천 엔진을 FastAPI로 감싼 서비스다.

지원 기능:

- 제품 목록 조회
- 제품 상세 조회
- `report_no` 기준 유사 제품 추천
- OCR 텍스트 기반 추천
- 원재료 직접 입력 기반 추천
- 이미지 업로드 기반 OCR 추천

추천 설명은 건강기능식품의 기능성원료, 표시 기능, 원재료 비교 기준으로만 생성하며, 질병 치료나 의약품 대체 표현은 사용하지 않는다.

## Install

```bash
pip install -r requirements_api.txt
```

## Runtime Files

- 제품 벡터 CSV: `c003_product_functional_vectors_final_rebuilt.csv`
- 제품 프로필 CSV: `output/product_function_profile.csv`
- SQLite DB: `ingredient_match_cache_rebuilt_item_class_i0050_final.sqlite`
- OCR source CSV: `건강기능식품_품목제조신고_원재료_C003.csv`

## OCR / LLM Config

기본 설정은 `api/config.py`와 기존 `config.json` 로딩을 함께 사용한다.

기본값:

- Google OCR credentials path: `D:\health_project\google_ocr_key`
- Local LLM endpoint: `http://169.213.5.157:3000/api/chat`
- Upload temp dir: `tmp/uploads`

권장 config 예시:

```json
{
  "google_ocr": {
    "enabled": true,
    "credentials_path": "D:\\health_project\\google_ocr_key"
  },
  "local_llm": {
    "enabled": true,
    "base_url": "http://169.213.5.157:3000/api/chat",
    "timeout_sec": 120,
    "max_retries": 2
  },
  "upload": {
    "temp_dir": "tmp/uploads",
    "max_file_size_mb": 10,
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".webp"]
  }
}
```

## Security Notes

- `google_ocr_key` 파일이나 JSON credentials 내용은 코드에 하드코딩하지 않는다.
- credentials 파일은 Git에 포함하지 않는다.
- 업로드 임시 파일은 `tmp/uploads/`에 저장 후 처리한다.

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
- `POST /api/recommend/by-ocr-text`
- `POST /api/recommend/by-image`

## Curl Examples

```bash
curl "http://localhost:8000/health"
```

```bash
curl "http://localhost:8000/api/catalog/products?page=1&page_size=20"
curl "http://localhost:8000/api/products/search?q=관절연골케어엔NEM"
curl "http://localhost:8000/api/products/2018001205671/profile"
curl "http://localhost:8000/api/products/2018001205671/similar?top_k=10&candidate_limit=1000"
```

```bash
curl -X POST "http://localhost:8000/api/recommend/by-ocr-text" ^
  -H "Content-Type: application/json" ^
  -d "{\"ocr_text\":\"제품명: 루테인 지아잔틴 플러스\n원재료명: 마리골드꽃추출물, 비타민A, 산화아연, 대두유, 밀납, 대두레시틴\n섭취량: 1일 1회, 1회 1캡슐\", \"top_k\":10, \"candidate_limit\":1000}"
```

```bash
curl -X POST "http://localhost:8000/api/recommend/by-ingredients" ^
  -H "Content-Type: application/json" ^
  -d "{\"ingredients\":[\"루테인\",\"지아잔틴\",\"비타민 A\",\"아연\"], \"top_k\":10, \"candidate_limit\":1000}"
```

```bash
curl -X POST "http://localhost:8000/api/recommend/by-image" ^
  -F "file=@sample_label.jpg" ^
  -F "top_k=10" ^
  -F "candidate_limit=1000"
```

## OCR Recommendation Flow

1. 이미지 업로드 또는 OCR 텍스트 입력
2. Google OCR로 텍스트 추출
3. OCR 텍스트에서 원재료 영역 파싱
4. 로컬 LLM으로 원료명 정규화 시도
5. 실패 시 rule-based fallback
6. `ingredient_match_cache` 기준 raw ingredient -> functional ingredient 매칭
7. 임시 제품 벡터 생성
8. 기존 제품 벡터와 유사도 계산
9. explanation 포함 TOP10 추천 반환

## needs_user_review

OCR 또는 LLM 파싱은 오인식 가능성이 있으므로 응답에 아래 필드가 항상 포함된다.

- `needs_user_review`
- `review_message`
- `parsed_ingredients_for_review`

프론트엔드는 이 값을 사용자에게 보여주고, 수정된 원재료 리스트를 다시 `POST /api/recommend/by-ingredients`로 보낼 수 있다.

## Cache Behavior

- `GET /api/products/{report_no}/similar`는 `product_similarity_explanation_cache`를 우선 사용한다.
- OCR/업로드 기반 추천은 입력이 매번 다를 수 있어 현재는 별도 영구 캐시를 강제하지 않는다.

## Duplicate Product Names

- 제품명 검색 API는 동일 제품명이 여러 `report_no`로 존재하면 모두 반환한다.
- 검색 단계에서 임의로 1개를 확정하지 않는다.
- 실제 추천은 사용자가 선택한 `report_no` 기준으로 호출한다.

## Safety Notice

기본 caution 문구:

> 본 추천은 건강기능식품의 기능성원료, 표시 기능, 원재료 정보를 기준으로 한 비교 결과입니다. 질병의 예방·치료 효과를 의미하지 않으며, 실제 섭취 전 제품 라벨의 함량, 1일 섭취량, 주의사항을 확인해야 합니다.

금지 표현:

- 치료
- 완치
- 의약품 대체
- 질병 개선
- 질환 개선
- 효능이 같다
- 효과가 같다
- 병을 낫게 한다
