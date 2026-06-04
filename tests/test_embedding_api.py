"""Tests for the embedding API endpoint."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from open_notebook.domain.notebook import Source


@pytest.fixture
def client():
    """Create test client after environment variables have been cleared by conftest."""
    from api.main import app

    return TestClient(app)


@pytest.mark.asyncio
@patch("api.routers.embedding.model_manager.get_embedding_model", new_callable=AsyncMock)
@patch("api.routers.embedding.Source.get", new_callable=AsyncMock)
async def test_embed_source_returns_400_for_missing_text(
    mock_source_get, mock_get_embedding_model, client
):
    """POST /embed should return 400 when the source has no text to vectorize."""
    mock_get_embedding_model.return_value = object()
    mock_source_get.return_value = Source(id="source:test", title="Test", full_text=None)

    response = client.post(
        "/api/embed",
        json={"item_id": "source:test", "item_type": "source"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Source source:test has no text to vectorize"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])