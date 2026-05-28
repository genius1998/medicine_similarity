# Deployment Guide

## 1. Prerequisites

- Python 3.10+ recommended
- Access to the SQLite runtime DB already used by the project
- Google OCR credentials file or directory
- Local GPT OSS endpoint if LLM parsing / reranking is enabled

## 2. Install

```powershell
cd D:\medicine_similarity_repo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_api.txt
```

## 3. Create Runtime Config

Copy the example file and create `config.json` in the repo root.

```powershell
Copy-Item .\config.example.json .\config.json
```

Required fields to update before deploy:

- `auth.default_admin_email`
- `auth.default_admin_password`
- `google_ocr.credentials_path`
- `local_llm.base_url`

Recommended production changes:

- `api.debug_response = false`
- `auth.cookie_secure = true` when running behind HTTPS

## 4. First Admin Account

On first startup, if there are no users in `app_user`, the app creates one admin user from:

- `auth.default_admin_email`
- `auth.default_admin_password`

This happens only once when the user table is empty.

Do not leave the example password in production.

## 5. Run

Development:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Production:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 6. Main Routes

- `/` : home
- `/login` : login
- `/signup` : signup
- `/products` : existing product DB list
- `/products/{report_no}` : existing product detail + similar products
- `/recommend/image` : OCR upload recommendation
- `/admin` : admin operations screen

## 7. Admin Features

Admin screen provides:

- monitoring summary
- OCR review history
- pending ingredient approval queue
- app event logs
- export of OCR reviews, pending ingredients, and logs

## 8. What Gets Stored

SQLite tables added for operations:

- `app_user`
- `app_session`
- `ocr_review_history`
- `ingredient_approval_queue`
- `app_event_log`

These are created automatically at startup.

## 9. Security Checklist

- set a strong admin password in `config.json`
- do not commit `config.json`
- keep Google OCR credentials outside Git
- enable HTTPS and set `auth.cookie_secure = true`
- restrict access to `/admin`
- rotate admin credentials after initial deploy if needed

## 10. Smoke Test

1. Open `/login`
2. Sign in with the configured admin account
3. Open `/admin`
4. Confirm monitoring page loads
5. Open `/products`
6. Open `/recommend/image`
7. Upload a sample OCR image and verify review history appears in `/admin`
