# OCR Image Recommendation Test Summary

## 1. 테스트 목적

건강기능식품 OCR 이미지 업로드 추천 파이프라인의 안정화 여부를 확인한다.

- Google OCR -> 로컬 LLM 파싱 -> 기능성원료 매칭 -> 임시 제품 벡터 생성 -> 유사 제품 TOP10 추천
- warning severity 분리
- summary / CSV / JSONL 집계 일치 여부 검증

## 2. 테스트 대상 이미지

- `product_001.png` ~ `product_012.jpg`

## 3. 실행 run_id

- `001_012_nutrition_fix_v2`

## 4. 테스트 결과 요약

- `total_images=12`
- `success_count=12`
- `failed_count=0`
- `recommendation_count_10_count=12`
- `needs_user_review_count=0`
- `critical_warning_count=0`
- `notice_count=152`
- `info_warning_count=12`

## 5. category_distribution

- 면역: 1
- 영양보충: 3
- 간 건강: 3
- 장 건강: 2
- 관절/연골: 1
- 혈당: 1
- 눈 건강: 1

## 6. top_similarity_stats

- `min=0.255964`
- `max=1.0`
- `avg=0.41663283333333334`

## 7. consistency check 결과

- `is_consistent=true`
- `summary / results CSV / results JSONL` 집계 일치

## 8. 주요 보정 내용

- OCR confidence `0.0` 고정 문제 해결
- `ocr_confidence_unavailable`을 `info`로 분리
- `allergen_notice`를 `notice`로 분리
- `HPMC` 등 부형제 primary 제외
- `난각막분말` / `NEM` 예외 처리
- 프로바이오틱스 strain canonical 압축
- 밀크씨슬 / 실리마린 간 건강 보정
- 혈당 / 바나바 / 키토산 규칙 보정
- 단일 강황 보조 성분이 관절/연골로 과잉 오버라이드되는 문제 완화

## 9. 남은 관찰 포인트

- `product_006`, `product_012`는 `top_similarity_score`가 낮은 편이므로 샘플 확장 시 관찰 필요
- `notice_count`가 많으므로 프론트에서는 알레르기/섭취주의를 별도 접기 영역으로 표시 권장
- Google Vision confidence가 `unavailable`로 내려오는 케이스가 계속 있으므로 confidence보다 파싱 품질/충돌 탐지를 우선 사용
