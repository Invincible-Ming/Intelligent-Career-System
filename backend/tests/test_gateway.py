"""Gateway configuration tests with synthetic files and no external services."""
import importlib.util
from pathlib import Path

import pytest
from dotenv import dotenv_values

spec = importlib.util.spec_from_file_location(
    "gateway_origin_config", Path(__file__).resolve().parents[2] / "deploy/gateway/configure_origin.py"
)
origin_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(origin_config)


def test_origin_configuration_preserves_other_values_and_is_idempotent(tmp_path):
    env = tmp_path / ".env"
    env.write_text("SYNTHETIC_KEY='fake-never-used'\nCORS_ORIGINS='http://localhost:8501'\n")
    origin = "https://career-test.trycloudflare.com"
    origin_config.allow_origin(origin + "/", env)
    origin_config.allow_origin(origin, env)
    config = dotenv_values(env)
    assert config["SYNTHETIC_KEY"] == "fake-never-used"
    allowed = config["CORS_ORIGINS"].split(",")
    assert "http://localhost:8501" in allowed
    assert "http://127.0.0.1:8080" in allowed
    assert allowed.count(origin) == 1


@pytest.mark.parametrize("origin", [
    "http://career-test.trycloudflare.com", "https://*.trycloudflare.com",
    "https://user:pass@career.example", "https://career.example/login",
    "https://career.example/?token=x", "https://career.example/#login",
    "https://career.example:99999", "file:///tmp/test", "career.example",
])
def test_invalid_origin_does_not_change_config(tmp_path, origin):
    env = tmp_path / ".env"
    original = "CORS_ORIGINS=http://localhost:8501\n"
    env.write_text(original)
    with pytest.raises(ValueError):
        origin_config.allow_origin(origin, env)
    assert env.read_text() == original


def test_gateway_browser_origin_boundary():
    from unittest.mock import patch

    from fastapi import HTTPException, Request

    from app.api.auth_api import check_browser_origin
    from app.core.config import settings

    public = "career-test.trycloudflare.com"
    origin = "https://" + public

    def request(host=public, source=origin):
        return Request({"type": "http", "method": "POST", "scheme": "https", "path": "/api/auth/browser-login",
                        "query_string": b"", "headers": [(b"host", host.encode()), (b"origin", source.encode())]})

    with patch.object(settings, "CORS_ORIGINS", origin):
        assert check_browser_origin(request(), origin + "/") == origin + "/"
        with pytest.raises(HTTPException) as error:
            check_browser_origin(request(source="https://attacker.invalid"), origin + "/")
        assert error.value.status_code == 403
        with pytest.raises(HTTPException) as error:
            check_browser_origin(request(host="localhost:8000"), origin + "/")
        assert error.value.status_code == 400
