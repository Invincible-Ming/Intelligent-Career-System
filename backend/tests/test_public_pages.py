"""Public rules use current config and approved return links, without a wallet/model call."""
from urllib.parse import urlencode

import httpx
import pytest
from fastapi import FastAPI

from app.api.public_pages import router
from app.core.config import settings


@pytest.fixture
async def pages(monkeypatch):
    monkeypatch.setattr(settings, "CORS_ORIGINS", "http://127.0.0.1:8501,http://127.0.0.1:8080,https://career.example,https://other.example")
    app = FastAPI()
    app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000") as client:
        yield client


@pytest.mark.anyio
async def test_public_rules_show_live_config_and_local_return_link(pages, monkeypatch):
    monkeypatch.setattr(settings, "CREDITS_INITIAL_POINTS", 240)
    monkeypatch.setattr(settings, "CREDITS_DAILY_CHECKIN_POINTS", 12)
    monkeypatch.setattr(settings, "CREDITS_UNITS_PER_TOKEN", 3)
    response = await pages.get("/pricing")
    assert response.status_code == 200
    assert 'class="stat-value">240' in response.text
    assert 'class="stat-value">+12' in response.text
    assert 'data-points-rate="3"' in response.text
    assert 'href="http://127.0.0.1:8501/?page=credits"' in response.text
    assert "__INITIAL_POINTS__" not in response.text and "__APP_URL__" not in response.text
    assert "no-store" in response.headers["cache-control"]


@pytest.mark.anyio
async def test_rules_preserve_public_origin_theme_and_credits_destination(pages):
    response = await pages.get("https://career.example/pricing", params={"return_to": "https://career.example/", "theme": "dark"})
    assert response.status_code == 200
    assert 'data-theme="dark"' in response.text
    assert 'href="https://career.example/?page=credits"' in response.text
    assert "127.0.0.1" not in response.text
    response = await pages.get("http://127.0.0.1:8080/pricing", params={"return_to": "http://127.0.0.1:8080/"})
    assert response.status_code == 200
    assert 'href="http://127.0.0.1:8080/?page=credits"' in response.text


@pytest.mark.anyio
@pytest.mark.parametrize("destination", [
    "https://other.example/", "https://attacker.example/", "//attacker.example/", "javascript:alert(1)",
    "https://career.example/?redirect=https://attacker.example/", "https://career.example/#injected",
    "https://career.example/path", "https://user:password@career.example/", "https://career.example:bad/",
])
async def test_rules_reject_unapproved_or_malformed_return_links(pages, destination):
    response = await pages.get("https://career.example/pricing", params={"return_to": destination})
    assert response.status_code == 400
    assert "password" not in response.text


@pytest.mark.anyio
async def test_rules_reject_invalid_theme(pages):
    response = await pages.get("/pricing?" + urlencode({"theme": '"><script>alert(1)</script>'}))
    assert response.status_code == 422


@pytest.mark.anyio
async def test_public_brand_images_are_explicitly_allowlisted(pages):
    response = await pages.get("/brand/logo-light.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")
    assert response.headers["x-content-type-options"] == "nosniff"
    for path in ("brand-spec.json", ".env", "app.py", "../app.py"):
        assert (await pages.get("/brand/" + path)).status_code == 404


@pytest.mark.anyio
async def test_rules_use_shared_brand_and_both_theme_logos(pages):
    content = (await pages.get("/pricing")).text
    assert "简面职说" in content
    assert "求职助手" not in content
    assert 'class="brand-light"' in content and 'class="brand-dark"' in content
    for asset in ("logo-light.png", "app-icon.png"):
        assert "/brand/" + asset in content
