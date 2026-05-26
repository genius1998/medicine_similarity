# Frontend Usage

## Overview

현재 프론트는 별도 React/Vite 앱이 아니라 **FastAPI + Jinja 템플릿 기반 MVP 화면**이다.

이 방식을 택한 이유:

- 저장소가 이미 FastAPI 템플릿 구조를 사용하고 있다.
- 이미지 업로드 -> OCR 결과 확인 -> 원료 수정 -> 재추천 흐름을 빠르게 검증하는 것이 우선이었다.
- 별도 프론트 빌드 체인을 추가하지 않아도 백엔드와 같은 서버에서 바로 테스트할 수 있다.

라우트:

- `http://localhost:8000/recommend/image`
- `http://localhost:8000/ocr-recommend`

두 라우트는 동일한 화면을 렌더링한다.

## Run

백엔드 서버를 먼저 실행한다.

```bash
cd D:\medicine_similarity_repo
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

그 다음 브라우저에서 아래 경로로 접속한다.

- `http://localhost:8000/recommend/image`

## API Base URL

- 기본값은 현재 페이지의 same-origin이다.
- 화면 상단의 `API Base URL` 입력란에서 다른 주소로 변경할 수 있다.
- 기본 개발 환경 예:
  - `http://localhost:8000`

## 화면 경로

- `/recommend/image`
- `/ocr-recommend`

## 사용 흐름

1. `/recommend/image` 페이지 접속
2. 성분표 이미지 업로드
3. `이미지 분석하기` 클릭
4. OCR 결과와 추정 카테고리, warning, 추천 TOP10 확인
5. 원료 리스트를 수정/삭제/추가
6. `수정한 원료로 다시 추천받기` 클릭
7. `/api/recommend/by-ingredients` 기준으로 추천 결과 갱신

## 테스트 이미지 예시

- `D:\health_project\input_images\product_001.png`

## 브라우저/HTML 검증 결과

브라우저 자동화 도구는 현재 환경에 설치되어 있지 않았다.

- `playwright`: 미설치
- `selenium`: 미설치

따라서 이번 검증은 아래 조합으로 수행했다.

- FastAPI 서버 실제 기동
- `GET /recommend/image` 200 확인
- `GET /ocr-recommend` 200 확인
- HTML 파싱으로 주요 버튼/입력란/섹션 존재 확인
- `by-image` API 실제 호출
- `by-ingredients` 재추천 API 실제 호출
- 두 라우트가 동일 화면 제목/단계 안내를 갖는지 비교

검증 스크립트:

```bash
python scripts/test_frontend_mvp_flow.py --base-url http://127.0.0.1:8000
```

생성 파일:

- `output/frontend_mvp_flow_test_summary.json`
- `output/frontend_mvp_flow_test_results.jsonl`

## 알려진 한계

- 실제 브라우저 클릭 자동화나 스크린샷 검증은 아직 없다.
- 현재 UI는 MVP이므로 복잡한 상태관리나 다중 페이지 라우팅은 포함하지 않는다.
- 추천 결과 카드는 API 응답 필드 누락에 대비해 fallback 처리를 하지만, 장기적으로는 프론트 타입 정의를 별도 분리하는 편이 낫다.

## 향후 React/Vite 분리 시점

아래 조건 중 하나가 되면 React/Vite 분리를 권장한다.

- 페이지 수가 늘어나고 라우팅이 복잡해질 때
- 로그인, 저장, 히스토리, 비교 UI 등 상태관리가 커질 때
- 디자인 시스템이나 공통 컴포넌트를 재사용해야 할 때
- OCR 수정 UI를 더 풍부하게 만들어야 할 때

## Git에 포함하면 안 되는 파일

- `.env`
- `.env.*`
- `google_ocr_key`
- `*google*key*`
- `*.credentials.json`
- `logs/`
- `output/*.csv`
- `output/*.jsonl`
- `output/failed_*.csv`
- `tmp/`
- `tmp/uploads/`

## 참고 사항

- 이 화면은 건강기능식품의 기능성원료/원재료 기준 비교를 위한 MVP UI다.
- 질병 예방·치료 효과를 의미하지 않는다.
- Google OCR key 파일이나 `.env` 파일은 Git에 포함하면 안 된다.
