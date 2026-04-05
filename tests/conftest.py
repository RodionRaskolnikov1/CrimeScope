"""
Shared pytest configuration and fixtures.
"""
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_model: mark test as requiring a trained model file in artifacts/models/"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip tests marked requires_model if no model file exists.
    This allows the test suite to run cleanly in CI without trained models.
    """
    from pathlib import Path
    model_path = Path("artifacts/models/crime_classifier.pkl")

    if not model_path.exists():
        skip_no_model = pytest.mark.skip(
            reason="No trained model found. Run 'uv run python main.py' first."
        )
        for item in items:
            if "requires_model" in item.keywords:
                item.add_marker(skip_no_model)