"""Small, independently testable boundaries shared by the MCP services."""
import base64
import ipaddress
import os
import stat
from pathlib import PurePosixPath
from urllib.parse import parse_qs, urlsplit

SEARCH_HOST = "www.bing.com"
DATABASE_ROLE = "career_mcp_reader"
VIEWS = ("knowledge_inventory", "evaluation_summary")
MAX_FILE_BYTES = 128 * 1024


def public_address(value):
    address = ipaddress.ip_address(value)
    return address.version == 4 and address.is_global and not address.is_multicast and not address.is_reserved


def validate_query(query, count):
    if any(ord(c) < 32 for c in query):
        raise ValueError("搜索词不能含控制字符")
    query = query.strip()
    if not query or len(query) > 300 or any(ord(c) < 32 for c in query):
        raise ValueError("搜索词必须为 1–300 字符，不含控制字符")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 5:
        raise ValueError("搜索结果数量必须为 1–5")
    return query


def result_url(value):
    """Unwrap Bing citation links; these URLs are returned, never fetched."""
    try:
        url = urlsplit(value)
        if url.hostname == SEARCH_HOST and url.path == "/ck/a":
            encoded = parse_qs(url.query).get("u", [""])[0]
            if encoded.startswith("a1"):
                value = base64.urlsafe_b64decode(encoded[2:] + "=" * (-len(encoded[2:]) % 4)).decode()
                url = urlsplit(value)
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password:
            return None
        hostname = url.hostname.lower()
        if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")) or "." not in hostname:
            return None
        try:
            if not ipaddress.ip_address(hostname).is_global:
                return None
        except ValueError:
            pass
        if url.port not in (None, 80, 443):
            return None
        return value
    except (ValueError, UnicodeError):
        return None


def path_parts(path):
    parts = PurePosixPath(path).parts
    if not parts or path.startswith("/") or "\\" in path or any(p.startswith(".") for p in parts):
        raise ValueError("仅允许专用目录内的相对路径，不允许隐藏文件或路径穿越")
    return parts


def open_workspace_path(root, path, *, directory=False):
    """Use openat + O_NOFOLLOW for every component, including raced symlinks."""
    parts = path_parts(path)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for index, part in enumerate(parts):
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if index < len(parts) - 1 or directory:
                flags |= os.O_DIRECTORY
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def read_workspace_file(root, path):
    fd = open_workspace_path(root, path)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
            raise ValueError("仅支持不超过 128 KiB 的普通 UTF-8 文本文件")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("文件超过读取上限")
        return data.decode("utf-8")
    finally:
        os.close(fd)
