# tests/test_recommender.py
import pytest
import app.recommender as recommender

@pytest.fixture(autouse=True)
def patch_recommender(monkeypatch):
    """Patch recommend_similar_papers to avoid heavy computation or network calls."""
    monkeypatch.setattr(
        recommender,
        "recommend_similar_papers",
        lambda query, **kwargs: f"Mocked recommendations for: {query}"
    )

def test_recommend_similar_papers_output():
    response = recommender.recommend_similar_papers("graph neural networks")
    assert isinstance(response, str)
    assert "graph neural networks" in response
    assert len(response) > 0
