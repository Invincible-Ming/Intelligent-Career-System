"""Public product information, rendered from the same rules used for billing."""
from html import escape
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse

from app.core.config import settings

router = APIRouter()
PRICING_HTML_PATH = Path(__file__).resolve().parents[3] / "frontend" / "projectflow-pricing.html"
BRAND_ASSET_PATH = PRICING_HTML_PATH.parent / "assets" / "brand"
BRAND_ASSETS = {name: BRAND_ASSET_PATH / name for name in (
    "mark-light.png", "logo-light.png", "app-icon.png",
)}


@router.get("/brand/{asset_name}", include_in_schema=False)
async def brand_asset(asset_name: str):
    path = BRAND_ASSETS.get(asset_name)
    if path is None or not path.is_file():
        raise HTTPException(404, "图片不存在")
    return FileResponse(path, media_type="image/png", headers={
        "Cache-Control": "public, max-age=3600", "X-Content-Type-Options": "nosniff",
    })


def app_return_url(request: Request, return_to: str | None) -> str:
    if return_to is None:
        if request.url.hostname in ("localhost", "127.0.0.1") and request.url.port == settings.APP_PORT:
            for port in (8501, 8080):
                origin = f"{request.url.scheme}://{request.url.hostname}:{port}"
                if origin in settings.cors_origin_list:
                    return origin + "/"
        return "/"
    try:
        parsed = urlsplit(return_to)
        parsed.port  # Reject malformed ports before constructing any link.
        origin = f"{parsed.scheme}://{parsed.netloc}"
        valid = (origin in settings.cors_origin_list and parsed.scheme in ("http", "https")
                 and parsed.hostname == request.url.hostname and parsed.path in ("", "/")
                 and not parsed.query and not parsed.fragment and not parsed.username and not parsed.password)
    except ValueError:
        valid = False
    if not valid:
        raise HTTPException(400, "返回应用地址必须是同域且已配置的前端地址")
    return origin + "/"


@router.get("/pricing", response_class=HTMLResponse, tags=["系统"], summary="积分规则页")
async def pricing_page(request: Request, return_to: str | None = Query(default=None, max_length=2048),
                       theme: Literal["auto", "light", "dark"] = "auto"):
    app_url = app_return_url(request, return_to)
    content = PRICING_HTML_PATH.read_text(encoding="utf-8")
    values = {"__INITIAL_POINTS__": settings.CREDITS_INITIAL_POINTS,
              "__DAILY_POINTS__": settings.CREDITS_DAILY_CHECKIN_POINTS,
              "__TOKEN_RATE__": settings.CREDITS_UNITS_PER_TOKEN,
              "__APP_URL__": app_url, "__CREDITS_URL__": app_url + "?page=credits", "__THEME__": theme}
    for marker, value in values.items():
        content = content.replace(marker, escape(str(value), quote=True))
    return HTMLResponse(content, headers={"Cache-Control": "no-store", "Referrer-Policy": "same-origin"})
