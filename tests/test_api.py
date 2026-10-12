"""
Tests for the FastAPI app.

/ask is mocked rather than calling the real pipeline -- keeps these tests
fast and runnable anywhere (CI, another machine) without needing Qdrant
or Azure OpenAI running. The real pipeline is already exercised for real
by the RAGAS evaluation script; these tests check the API layer's own
behavior (routing, validation, response shape).
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ask_rejects_too_short_question():
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 422


def test_ask_rejects_missing_question():
    response = client.post("/ask", json={})
    assert response.status_code == 422


@patch("src.api.app.answer_question")
def test_ask_returns_expected_shape_on_success(mock_answer_question):
    mock_answer_question.return_value = {
        "question": "Test question?",
        "answer": "Test answer.",
        "contexts": ["some context"],
        "sources": [
            {"text": "some context", "source_file": "test.pdf", "page_number": 1, "score": 0.9}
        ],
    }

    response = client.post("/ask", json={"question": "Test question?", "top_k": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "Test question?"
    assert body["answer"] == "Test answer."
    assert len(body["sources"]) == 1
    assert body["sources"][0]["source_file"] == "test.pdf"
    assert body["sources"][0]["page_number"] == 1


@patch("src.api.app.answer_question")
def test_ask_returns_500_on_pipeline_failure(mock_answer_question):
    mock_answer_question.side_effect = Exception("Qdrant connection failed")

    response = client.post("/ask", json={"question": "Test question?"})

    assert response.status_code == 500
    assert "Failed to generate an answer" in response.json()["detail"]
    