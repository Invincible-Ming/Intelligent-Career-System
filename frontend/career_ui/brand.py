"""The public identity shared by the app's entry points."""
from html import escape
from pathlib import Path

from career_ui.links import public_backend_base

NAME = "简面职说"
DESCRIPTOR = "大学生的 AI 求职搭子"
TAGLINE = "简历怎么写，面试怎么说。"
INTRO = "从梳理简历、读懂岗位，到模拟面试、规划学习，陪你把求职准备一步步做好。"
ASSETS = Path(__file__).resolve().parents[1] / "assets" / "brand"
APP_ICON = ASSETS / "app-icon.png"


def brand_html(css_class: str = "product-wordmark") -> str:
    base = escape(public_backend_base(), quote=True)
    return (f'<div class="{escape(css_class, quote=True)} brand-lockup" role="img" aria-label="{NAME}">'
            f'<img class="brand-light" src="{base}/brand/logo-light.png?v=1" alt="" width="2172" height="724">'
            f'<img class="brand-dark" src="{base}/brand/logo-light.png?v=1" alt="" width="2172" height="724">'
            '</div>')
