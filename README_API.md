# API Usage

## Overview

이 프로젝트는 건강기능식품 추천 API다. 제품명/`report_no` 기반 추천과 OCR 이미지 업로드 기반 추천을 모두 지원한다.

주요 기능:

- 제품 검색
- 제품 프로필 조회
- `report_no` 기반 유사 제품 추천
- OCR 텍스트 기반 추천
- 원재료 리스트 기반 추천
- 이미지 업로드 기반 OCR 추천

추천 결과는 건강기능식품의 기능성원료, 표시 기능, 원재료 정보를 기준으로 비교한 결과이며 의학적 진단이나 치료 판단을 의미하지 않는다.

## Install

```bash
pip install -r requirements_api.txt
```

## Run

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Google OCR 설정

- Google Vision credentials path는 `config` 또는 환경변수로 지정한다.
- 개발 환경 기본 경로 예:
  - `D:\health_project\google_ocr_key`
- key 파일은 Git에 포함하면 안 된다.
- key 경로를 코드에 하드코딩하지 말고 설정값으로 주입해야 한다.

## 로컬 LLM 설정

- 기본 endpoint:
  - `http://169.213.5.157:3000/api/chat`
- 설정 파일 또는 환경변수로 변경 가능해야 한다.
- OCR 텍스트에서 원재료 영역 파싱, 정규화, warning 보정에 사용한다.

## 주요 API 엔드포인트

- `GET /health`
- `GET /api/products/search?q=검색어&limit=20`
- `GET /api/products/{report_no}/profile`
- `GET /api/products/{report_no}/similar?top_k=10&candidate_limit=1000&force_refresh=false`
- `POST /api/recommend/by-image`
- `POST /api/recommend/by-ocr-text`
- `POST /api/recommend/by-ingredients`

## by-image 예시

```bash
curl -X POST "http://localhost:8000/api/recommend/by-image" ^
  -F "file=@sample_label.jpg" ^
  -F "top_k=10" ^
  -F "candidate_limit=1000"
```

## by-ocr-text 예시

```bash
curl -X POST "http://localhost:8000/api/recommend/by-ocr-text" ^
  -H "Content-Type: application/json" ^
  -d "{\"ocr_text\":\"제품명: 루테인 지아잔틴 플러스\n원재료명: 마리골드꽃추출물, 비타민A, 산화아연\", \"top_k\":10, \"candidate_limit\":1000}"
```

## by-ingredients 예시

```bash
curl -X POST "http://localhost:8000/api/recommend/by-ingredients" ^
  -H "Content-Type: application/json" ^
  -d "{\"ingredients\":[\"루테인\",\"지아잔틴\",\"비타민 A\",\"아연\"], \"top_k\":10}"
```

## OCR 응답 품질 필드

- `needs_user_review`
- `review_message`
- `critical_warnings`
- `notices`
- `info_warnings`
- `parsed_ingredients_for_review`
- `excluded_ingredients`
- `primary_ingredients_normalized`
- `estimated_profile`

## warning severity

`critical_warnings`

- 실제 추천 품질 문제
- 카테고리 충돌
- 원료 파싱 불확실
- 추천 유사도 낮음
- primary 불명확

`notices`

- 알레르기 안내
- 섭취 주의
- 제조시설 안내
- 보관방법
- 고객센터/제조원/판매원 안내

`info_warnings`

- `ocr_confidence_unavailable`
- OCR confidence 정보 없음
- 로그성 안내

## OCR 추천 흐름

1. 이미지 업로드 또는 OCR 텍스트 입력
2. Google OCR로 텍스트 추출
3. 로컬 LLM으로 원재료 영역 파싱
4. 기능성 원료 정규화 및 canonical 매칭
5. 임시 제품 벡터 생성
6. 기존 제품 벡터와 유사도 계산
7. explanation 포함 TOP10 추천 반환

## Safety Caution

API 응답에는 반드시 아래 caution 문구를 포함해야 한다.

> 본 추천은 건강기능식품의 기능성원료, 표시 기능, 원재료 정보를 기준으로 한 비교 결과입니다. 질병의 예방·치료 효과를 의미하지 않으며, 실제 섭취 전 제품 라벨의 함량, 1일 섭취량, 주의사항을 확인해야 합니다.

## 금지 표현

`reason` 또는 `explanation`에 아래 표현을 사용하면 안 된다.

- 치료
- 완치
- 의약품 대체
- 질병 개선
- 질환 개선
- 효능이 같다
- 효과가 같다
- 병을 낫게 한다
