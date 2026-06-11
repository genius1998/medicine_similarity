# API Endpoint Test Summary

## 1. 테스트 일시

- 2026-05-16 21:35 KST

## 2. base_url

- `http://127.0.0.1:8000`

## 3. 테스트한 엔드포인트 목록

- `GET /health`
- `GET /api/products/search?q=관절연골케어엔NEM&limit=20`
- `GET /api/products/search?q=인지력 건강엔 포스파티딜세린&limit=20`
- `GET /api/products/2018001205671/profile`
- `GET /api/products/2018001205671/similar?top_k=10&candidate_limit=1000&force_refresh=true`
- `GET /api/products/2018001205671/similar?top_k=10&candidate_limit=1000&force_refresh=false`
- `POST /api/recommend/by-ocr-text`
- `POST /api/recommend/by-ingredients` structured input
- `POST /api/recommend/by-ingredients` raw ingredient text
- `POST /api/recommend/by-image`

## 4. 성공/실패 요약

- `total_tests=10`
- `passed=10`
- `failed=0`

## 5. 제품명 검색 결과 요약

- `관절연골케어엔NEM` 검색은 `report_no`, `product_name`, `product_main_category`, `primary_ingredients`를 포함한 결과 배열을 정상 반환했다.
- `인지력 건강엔 포스파티딜세린` 검색은 동일 제품명 다건 반환이 유지됐다.
- 검색 API는 임의로 1건을 확정하지 않는다.

## 6. 유사 제품 추천 결과 요약

- `GET /api/products/2018001205671/similar`
- `base_product` 정상 반환
- `recommendations=10`
- 각 추천 항목에 `similarity_score`, `substitutability`, `reason`, `caution`, `explanation` 존재
- 상위 추천은 `난각막가수분해물(NEM®)` 기반 관절/연골 제품으로 정렬됐다.

## 7. OCR 텍스트 추천 결과 요약

- 입력 예시: 루테인/지아잔틴/비타민A/아연
- `estimated_profile.product_main_category=눈 건강`
- `recommendations=10`
- `needs_user_review=false`
- `critical_warnings=[]`
- `info_warnings=[]`

## 8. 이미지 업로드 추천 결과 요약

- 테스트 이미지: `input_images/product_001.png`
- `ocr.raw_text` 존재
- `parsed` 존재
- `estimated_profile.product_main_category=면역`
- `recommendations=10`
- `needs_user_review=false`
- `critical_warnings=[]`
- `notices`와 `info_warnings` 분리 정상

## 9. 캐시 동작 확인 여부

- 확인됨
- 1차 호출
  - `cache_used=false`
  - 서버 실행시간 약 `1.798730s`
- 2차 호출
  - `cache_used=true`
  - 서버 실행시간 약 `0.024321s`

## 10. 프론트 연결 시 사용할 주요 필드

공통 추천 응답:

- `base_product` 또는 `estimated_profile`
- `recommendations`
- `recommendations[].rank`
- `recommendations[].target_report_no`
- `recommendations[].target_product_name`
- `recommendations[].similarity_score`
- `recommendations[].substitutability`
- `recommendations[].shared_ingredients`
- `recommendations[].shared_categories`
- `recommendations[].reason`
- `recommendations[].caution`

OCR 기반 응답:

- `needs_user_review`
- `review_message`
- `parsed_ingredients_for_review`
- `critical_warnings`
- `notices`
- `info_warnings`
- `excluded_ingredients`
- `estimated_profile`

## 11. 남은 이슈

- `by-ingredients`는 이번 수정으로 OCR warning을 타지 않도록 분리됐다.
- `by-ocr-text`는 OCR confidence가 없더라도 현재는 review를 강제하지 않으며, 입력 텍스트 기반 파싱 품질만으로 동작한다.
- 콘솔/터미널 출력에서는 일부 한글이 깨져 보일 수 있다. 응답 JSON과 저장 파일은 정상 UTF-8 기준으로 확인됐다.
