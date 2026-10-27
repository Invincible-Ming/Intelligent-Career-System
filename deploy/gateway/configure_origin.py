"""Allow one exact browser origin without reading credentials into output."""
import argparse
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values, set_key

ENV_FILE = Path(__file__).resolve().parents[2] / "backend" / ".env"


def normalize_origin(value: str) -> str:
    parsed = urlsplit(value.strip())
    # Accessing port also validates malformed/out-of-range port numbers.
    parsed.port
    if (parsed.scheme not in ("https", "http") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("请输入完整域名来源，例如 https://example.r9.cpolar.top，不要附带路径或参数")
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("公网入口必须使用 HTTPS；HTTP 仅限本机开发")
    if "*" in parsed.netloc:
        raise ValueError("请使用本次隧道的准确域名，不允许通配符")
    return f"{parsed.scheme}://{parsed.netloc.lower()}"


def allow_origin(value: str, env_file: Path = ENV_FILE) -> str:
    origin = normalize_origin(value)
    if not env_file.is_file():
        raise ValueError(f"配置文件不存在：{env_file}")
    current = dotenv_values(env_file).get("CORS_ORIGINS")
    origins = [item.strip() for item in (current or "http://localhost:8501,http://127.0.0.1:8501").split(",")
               if item.strip()]
    for item in ("http://localhost:8080", "http://127.0.0.1:8080", origin):
        if item not in origins:
            origins.append(item)
    set_key(env_file, "CORS_ORIGINS", ",".join(origins))
    return origin


def main():
    parser = argparse.ArgumentParser(description="将当前隧道的准确域名加入后端登录允许列表")
    parser.add_argument("origin", help="cpolar 输出的 HTTPS URL")
    args = parser.parse_args()
    try:
        origin = allow_origin(args.origin)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"已允许：{origin}")
    print("请重启后端使 CORS_ORIGINS 生效；前端将自动使用访问时的公网域名。")


if __name__ == "__main__":
    main()
