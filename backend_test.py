import pytest
from fastapi.testclient import TestClient
from main import app, init_db, Base, engine

# Setup test DB
Base.metadata.drop_all(bind=engine)
init_db()

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "provider" in response.json()

def test_translate_mock_success():
    payload = {
        "text": "Hello world",
        "source_lang": "en",
        "target_lang": "vi"
    }
    response = client.post("/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "MOCK VI" in data["translated_text"]
    assert data["provider"] == "mock"

def test_translate_validation_length():
    # Test text too long
    long_text = "a" * 501
    payload = {
        "text": long_text,
        "source_lang": "en",
        "target_lang": "vi"
    }
    response = client.post("/translate", json=payload)
    assert response.status_code == 422 # Unprocessable Entity

def test_translate_with_glossary():
    payload = {
        "text": "I love Apple products",
        "source_lang": "en",
        "target_lang": "vi",
        "glossary": {
            "Apple": "Táo Khuyết"
        }
    }
    response = client.post("/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Với Mock adapter, nó sẽ replace string đơn giản
    assert "Táo Khuyết" in data["translated_text"] or data["glossary_applied"] == True

def test_history_logging():
    # Make a request
    client.post("/translate", json={"text": "Log me", "source_lang": "en", "target_lang": "fr"})
    
    # Check history
    response = client.get("/history")
    assert response.status_code == 200
    history = response.json()
    assert len(history) > 0
    assert history[0]["source_text"] == "Log me"