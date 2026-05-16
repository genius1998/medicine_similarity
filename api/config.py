from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ApiSettings:
    service_name: str = "health-functional-product-recommender"
    root_dir: Path = Path(__file__).resolve().parents[1]
    default_top_k: int = 10
    default_candidate_limit: int = 1000
    default_max_df_for_seed: int = 3000


def get_settings() -> ApiSettings:
    return ApiSettings()
