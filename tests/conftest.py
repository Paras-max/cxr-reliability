from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def v0_thresholds_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "thresholds" / "v0_prd_defaults.yaml"


@pytest.fixture(scope="session")
def pipeline_config_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "pipeline.yaml"
