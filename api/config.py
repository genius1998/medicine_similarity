from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from config_loader import get_config_value, load_config


@dataclass(frozen=True)
class ApiSettings:
    service_name: str
    root_dir: Path
    default_top_k: int
    default_candidate_limit: int
    default_max_df_for_seed: int
    catalog_csv_name_pattern: str
    templates_dir: Path
    google_ocr_enabled: bool
    google_ocr_credentials_path: Path
    local_llm_enabled: bool
    local_llm_base_url: str
    local_llm_timeout_sec: int
    local_llm_max_retries: int
    upload_temp_dir: Path
    upload_max_file_size_mb: int
    upload_allowed_extensions: Tuple[str, ...]
    debug_response: bool
    auth_default_admin_email: str
    auth_default_admin_password: str
    auth_session_cookie_name: str
    auth_session_hours: int
    auth_cookie_secure: bool
    ingredient_embedding_enabled: bool
    ingredient_embedding_model_name: str
    ingredient_embedding_metadata_path: Path
    ingredient_embedding_documents_path: Path
    ingredient_embedding_cache_path: Path


def _nested_value(config: dict, section: str, key: str, default):
    section_value = config.get(section, {})
    if not isinstance(section_value, dict):
        return default
    value = section_value.get(key, default)
    return default if value in ("", None) else value


def get_settings() -> ApiSettings:
    root_dir = Path(__file__).resolve().parents[1]
    config = load_config()

    service_name = str(get_config_value(config, "api_service_name", "health-functional-product-recommender"))
    default_top_k = int(get_config_value(config, "api_default_top_k", 10))
    default_candidate_limit = int(get_config_value(config, "api_default_candidate_limit", 1000))
    default_max_df_for_seed = int(get_config_value(config, "api_default_max_df_for_seed", 3000))

    google_ocr_enabled = bool(_nested_value(config, "google_ocr", "enabled", True))
    google_ocr_credentials_path = Path(
        str(_nested_value(config, "google_ocr", "credentials_path", r"D:\health_project\google_ocr_key"))
    )

    local_llm_enabled = bool(_nested_value(config, "local_llm", "enabled", True))
    local_llm_base_url = str(_nested_value(config, "local_llm", "base_url", "http://169.213.5.157:3000/api/chat"))
    local_llm_timeout_sec = int(_nested_value(config, "local_llm", "timeout_sec", 120))
    local_llm_max_retries = int(_nested_value(config, "local_llm", "max_retries", 2))

    upload_temp_dir = root_dir / str(_nested_value(config, "upload", "temp_dir", "tmp/uploads"))
    upload_max_file_size_mb = int(_nested_value(config, "upload", "max_file_size_mb", 10))
    allowed = _nested_value(config, "upload", "allowed_extensions", [".jpg", ".jpeg", ".png", ".webp"])
    upload_allowed_extensions = tuple(str(item).lower() for item in allowed)
    debug_response_value = os.getenv("DEBUG_RESPONSE", _nested_value(config, "api", "debug_response", True))
    debug_response = str(debug_response_value).strip().lower() not in {"0", "false", "off", "no", ""}
    auth_default_admin_email = str(os.getenv("APP_DEFAULT_ADMIN_EMAIL", _nested_value(config, "auth", "default_admin_email", "admin@example.com")))
    auth_default_admin_password = str(os.getenv("APP_DEFAULT_ADMIN_PASSWORD", _nested_value(config, "auth", "default_admin_password", "")))
    auth_session_cookie_name = str(os.getenv("APP_SESSION_COOKIE_NAME", _nested_value(config, "auth", "session_cookie_name", "sb_session")))
    auth_session_hours = int(os.getenv("APP_SESSION_HOURS", _nested_value(config, "auth", "session_hours", 12)))
    auth_cookie_secure_value = os.getenv("APP_COOKIE_SECURE", _nested_value(config, "auth", "cookie_secure", False))
    auth_cookie_secure = str(auth_cookie_secure_value).strip().lower() in {"1", "true", "on", "yes"}
    ingredient_embedding_enabled_value = os.getenv("INGREDIENT_EMBEDDING_ENABLED", _nested_value(config, "ingredient_embedding", "enabled", True))
    ingredient_embedding_enabled = str(ingredient_embedding_enabled_value).strip().lower() not in {"0", "false", "off", "no", ""}
    ingredient_embedding_model_name = str(
        os.getenv(
            "INGREDIENT_EMBEDDING_MODEL_NAME",
            _nested_value(config, "ingredient_embedding", "model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        )
    )
    ingredient_embedding_metadata_path = Path(
        str(
            os.getenv(
                "INGREDIENT_EMBEDDING_METADATA_PATH",
                _nested_value(config, "ingredient_embedding", "metadata_path", r"D:\db\deploy_ec2\data\functional_ingredient_metadata_item_class_final_boosted.pkl"),
            )
        )
    )
    ingredient_embedding_documents_path = Path(
        str(
            os.getenv(
                "INGREDIENT_EMBEDDING_DOCUMENTS_PATH",
                _nested_value(config, "ingredient_embedding", "documents_path", r"D:\db\deploy_ec2\data\functional_ingredient_rag_documents_merged_item_class_boosted.csv"),
            )
        )
    )
    ingredient_embedding_cache_path = root_dir / str(
        _nested_value(config, "ingredient_embedding", "cache_path", "tmp/ingredient_embedding_cache")
    )

    templates_dir = root_dir / "api" / "templates"
    upload_temp_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "logs").mkdir(parents=True, exist_ok=True)
    ingredient_embedding_cache_path.mkdir(parents=True, exist_ok=True)
    return ApiSettings(
        service_name=service_name,
        root_dir=root_dir,
        default_top_k=default_top_k,
        default_candidate_limit=default_candidate_limit,
        default_max_df_for_seed=default_max_df_for_seed,
        catalog_csv_name_pattern="*C003.csv",
        templates_dir=templates_dir,
        google_ocr_enabled=google_ocr_enabled,
        google_ocr_credentials_path=google_ocr_credentials_path,
        local_llm_enabled=local_llm_enabled,
        local_llm_base_url=local_llm_base_url,
        local_llm_timeout_sec=local_llm_timeout_sec,
        local_llm_max_retries=local_llm_max_retries,
        upload_temp_dir=upload_temp_dir,
        upload_max_file_size_mb=upload_max_file_size_mb,
        upload_allowed_extensions=upload_allowed_extensions,
        debug_response=debug_response,
        auth_default_admin_email=auth_default_admin_email,
        auth_default_admin_password=auth_default_admin_password,
        auth_session_cookie_name=auth_session_cookie_name,
        auth_session_hours=auth_session_hours,
        auth_cookie_secure=auth_cookie_secure,
        ingredient_embedding_enabled=ingredient_embedding_enabled,
        ingredient_embedding_model_name=ingredient_embedding_model_name,
        ingredient_embedding_metadata_path=ingredient_embedding_metadata_path,
        ingredient_embedding_documents_path=ingredient_embedding_documents_path,
        ingredient_embedding_cache_path=ingredient_embedding_cache_path,
    )
