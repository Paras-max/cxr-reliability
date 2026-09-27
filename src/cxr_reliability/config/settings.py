"""Environment-driven runtime settings.

Responsibility:
    Read CXR_* environment variables / .env into one validated object: device, seed,
    directories, active thresholds version, API and tracking settings.

Input:
    .env file and process environment.

Output:
    Settings (pydantic BaseSettings); get_settings() returns a cached instance.

Dependencies:
    pydantic-settings

Implementation phase: P0
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CXR_", env_file=".env", extra="ignore")

    env: str = "dev"
    device: str = "cpu"
    seed: int = 42

    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    outputs_dir: Path = Path("outputs")
    logs_dir: Path = Path("logs")
    configs_dir: Path = Path("configs")

    # NIH ChestX-ray14 dataset root.  Override with CXR_DATASET_ROOT in .env
    # or as an environment variable.  Defaults to the dataset/ folder that
    # lives alongside the project root (one level up from cwd).
    dataset_root: Path = Path("dataset")

    thresholds_version: str = "v0_prd_defaults"
    pipeline_config: Path = Path("configs/pipeline.yaml")

    mlflow_tracking_uri: str = "file:./outputs/mlruns"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_upload_mb: int = 20

    @property
    def thresholds_path(self) -> Path:
        return self.configs_dir / "thresholds" / f"{self.thresholds_version}.yaml"


@lru_cache
def get_settings() -> Settings:
    return Settings()
