import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import httpx
from httpx import ASGITransport
from unittest.mock import AsyncMock, patch

BASE_URL = "http://localhost:8000"
TEST_QUERIES = [
    "How do I reverse a list in Python?",
    "How do I read a CSV file with pandas?",
    "What is the difference between __str__ and __repr__?",
    "How do I merge two dictionaries in Python 3?",
    "How do I use @property decorator in Python?",
    "What is the difference between asyncio.gather and asyncio.wait?",
    "How do I profile Python code to find bottlenecks?",
    "What is Python?",
    "How do I unpack nested list comprehensions with walrus operator?",
    "How do I install Node.js packages?",
]
MOCK_RESPONSE = {"question":"test","answer":"Use lst[::-1]","sources":[{"title":"Reverse","score":10,"relevance":0.92,"so_id":"1"}],"latency_ms":120,"model":"llama-3.1-8b-instant"}

@pytest.fixture
def mock_rag():
    with patch("app.routers.qa.rag_pipeline") as mock:
        mock.ready = True
        mock.ask = AsyncMock(return_value=MOCK_RESPONSE)
        yield mock

@pytest.fixture
def mock_health_rag():
    with patch("app.routers.health.rag_pipeline") as mock:
        mock.ready = True
        mock.stats.return_value = {"status":"ready","documents":50000,"model":"llama-3.1-8b-instant","embedding":"all-MiniLM-L6-v2","top_k":5}
        yield mock

@pytest.mark.asyncio
async def test_health_endpoint(mock_health_rag):
    from app.main import app
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/health")
    assert r.status_code == 200
    assert "documents" in r.json()

@pytest.mark.asyncio
async def test_ask_happy_path(mock_rag):
    from app.main import app
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post("/ask", json={"question": "How do I reverse a list?"})
    assert r.status_code == 200
    assert "answer" in r.json()

@pytest.mark.asyncio
async def test_ask_empty_question():
    from app.main import app
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post("/ask", json={"question": ""})
    assert r.status_code == 422

@pytest.mark.asyncio
async def test_ask_too_short_question():
    from app.main import app
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post("/ask", json={"question": "hi"})
    assert r.status_code == 422

@pytest.mark.asyncio
async def test_ask_missing_field():
    from app.main import app
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post("/ask", json={})
    assert r.status_code == 422

@pytest.mark.asyncio
async def test_ask_very_long_question(mock_rag):
    from app.main import app
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.post("/ask", json={"question": "How do I " + "really " * 200 + "use Python?"})
    assert r.status_code in (200, 422)

@pytest.mark.asyncio
async def test_service_unavailable_when_not_ready():
    from app.main import app
    with patch("app.routers.qa.rag_pipeline") as mock_pipe:
        mock_pipe.ready = False
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.post("/ask", json={"question": "How do I use pandas?"})
    assert r.status_code == 503

@pytest.mark.asyncio
async def test_root_endpoint():
    from app.main import app
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/")
    assert r.status_code == 200

@pytest.mark.integration
def test_live_health():
    r = httpx.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200

@pytest.mark.integration
@pytest.mark.parametrize("question", TEST_QUERIES)
def test_live_ask(question):
    r = httpx.post(f"{BASE_URL}/ask", json={"question": question}, timeout=30)
    assert r.status_code == 200 and len(r.json()["answer"]) > 10
