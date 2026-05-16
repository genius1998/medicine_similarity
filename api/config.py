from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ApiSettings:
    service_name: str = "health-functional-product-recommender"
    root_dir: Path = Path(__file__).resolve().parents[1]
    default_top_k: int = 10
    default_candidate_limit: int = 1000
    default_max_df_for_seed: int = 3000
    catalog_csv_name_pattern: str = "*C003.csv"
    templates_dir: Optional[Path] = None


def get_settings() -> ApiSettings:
    settings = ApiSettings()
    object.__setattr__(settings, "templates_dir", settings.root_dir / "api" / "templates")
    return settings
