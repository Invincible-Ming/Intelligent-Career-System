"""后端 API 客户端。

从 app.py 抽出，保持行为完全一致：
- 统一鉴权头、超时策略与错误包装；
- SSE 流式接口逐事件产出；
- 业务接口的薄封装。

原先依赖 app.py 的模块级全局 security_options（通过 globals() 读取），
这里改为显式参数传入，避免隐式耦合。
"""
from __future__ import annotations

import json
from typing import Any, Iterator

import requests
import streamlit as st

from career_ui.config import API_BASE_URL, LONG_TIMEOUT, SHORT_TIMEOUT

# 服务端下发的限制参数，登录后由 fetch_security_options() 填充。
# 键名与后端 /api/auth/options 的响应一致；缺失时回落到保守默认值。
DEFAULT_SECURITY_OPTIONS: dict[str, Any] = {
    "knowledge_provider": "milvus",
    "registration_enabled": False,
    "max_upload_size_mb": 20,
    "chat_max_input_chars": 4000,
    "chat_max_output_tokens": 2048,
    "chat_total_timeout": 90,
    "analysis_total_timeout": 180,
    "match_total_timeout": 600,
    "upload_total_timeout": 120,
    "api_total_timeout": 30,
    "search_max_input_chars": 2000,
    "jd_max_input_chars": 30000,
    "feedback_max_input_chars": 2000,
    "auth_cookie_name": "career_session",
}

# 当前会话生效的限制参数。由 app.py 在登录后设置。
_security_options: dict[str, Any] = dict(DEFAULT_SECURITY_OPTIONS)


class APIError(Exception):
    """后端 API 请求异常。"""


def security_options() -> dict[str, Any]:
    """返回当前生效的服务端限制参数。"""

    return _security_options


def set_security_options(values: dict[str, Any] | None) -> None:
    """用服务端返回值覆盖限制参数；未登录时保持默认值。"""

    global _security_options

    if values:
        _security_options = {**DEFAULT_SECURITY_OPTIONS, **values}
    else:
        _security_options = dict(DEFAULT_SECURITY_OPTIONS)


def auth_headers() -> dict[str, str]:
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def operation_read_timeout(path: str) -> int:
    """按接口类型选择读超时。

    与后端 limits.operation_timeout 对齐，并留 10 秒余量给网络与序列化。
    """

    limits = _security_options
    if "/chat/completions" in path:
        return limits["chat_total_timeout"] + 10
    if "/documents/upload" in path or (path.startswith("/api/documents/") and path.endswith("/retry")):
        return limits["upload_total_timeout"] + 10
    if "/match" in path:
        return limits["match_total_timeout"] + 10
    if any(part in path for part in ("/interview", "/learning-plan", "/evaluation")):
        return limits["analysis_total_timeout"] + 10
    return limits["api_total_timeout"] + 10


def request_api(
    method: str,
    path: str,
    *,
    timeout: int = SHORT_TIMEOUT,
    raw: bool = False,
    access_token: str | None = None,
    **kwargs: Any,
) -> Any:
    """调用 FastAPI，并统一处理错误。"""

    # Deferred downloads run outside the Streamlit script thread. Their token
    # is captured by the owning session; the backend revalidates it on click.
    headers = {"Authorization": f"Bearer {access_token}"} if access_token is not None else auth_headers()
    kwargs["headers"] = {**kwargs.pop("headers", {}), **headers}
    url = f"{API_BASE_URL}{path}"

    try:
        response = requests.request(
            method=method,
            url=url,
            timeout=(5, operation_read_timeout(path) if timeout == LONG_TIMEOUT else timeout),
            **kwargs,
        )
    except requests.ConnectionError as exc:
        raise APIError(f"无法连接后端服务：{API_BASE_URL}") from exc
    except requests.Timeout as exc:
        raise APIError("请求超时，请检查后端日志后重试") from exc
    except requests.RequestException as exc:
        raise APIError(f"请求失败：{exc}") from exc

    if not response.ok:
        if response.status_code == 401 and path != "/api/auth/login" and access_token is None:
            from career_ui.session import reset_login_state

            reset_login_state()
        try:
            error_data = response.json()
            detail = error_data.get("detail", error_data)
        except ValueError:
            detail = response.text or response.reason

        raise APIError(f"HTTP {response.status_code}：{detail}")

    if raw:
        return response.content
    if not response.content:
        return None

    try:
        return response.json()
    except ValueError as exc:
        raise APIError("后端返回了无法解析的数据") from exc


def _open_stream(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_message: str,
) -> requests.Response:
    """建立 SSE 连接并校验响应头。"""

    try:
        response = requests.post(
            url,
            json=payload,
            stream=True,
            headers=auth_headers(),
            timeout=(5, operation_read_timeout(url)),
        )
    except requests.ConnectionError as exc:
        raise APIError(f"无法连接后端服务：{API_BASE_URL}") from exc
    except requests.Timeout as exc:
        raise APIError(timeout_message) from exc
    except requests.RequestException as exc:
        raise APIError(f"请求失败：{exc}") from exc

    if response.status_code == 401:
        response.close()
        from career_ui.session import reset_login_state

        reset_login_state()
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text or response.reason
        response.close()
        raise APIError(f"HTTP {response.status_code}：{detail}")

    return response


def _iter_sse(
    response: requests.Response,
    *,
    chunk_size: int | None,
) -> Iterator[tuple[str, str]]:
    """把 SSE 字节流拆成 (event_type, data_str)。"""

    current_event = "message"
    for line in response.iter_lines(chunk_size=chunk_size, decode_unicode=True):
        if not line:
            continue
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            yield current_event, line[len("data:"):].strip()


def stream_chat_api(
    prompt: str,
    conversation_id: str | None = None,
) -> Iterator[str]:
    """调用智能对话 SSE 流式接口，逐 Token 产出文字供打字机渲染。"""

    payload: dict[str, Any] = {"message": prompt, "stream": True}
    if conversation_id:
        payload["conversation_id"] = conversation_id

    response = _open_stream(
        f"{API_BASE_URL}/api/chat/completions",
        payload,
        timeout_message="流式对话连接超时",
    )

    try:
        for _event, data_str in _iter_sse(response, chunk_size=None):
            if data_str == "[DONE]":
                break
            try:
                data_json = json.loads(data_str)
            except json.JSONDecodeError:
                if data_str:
                    yield data_str
                continue

            if "error" in data_json:
                raise APIError(
                    data_json["error"].get("message", "回复生成已中断")
                )

            if "conversation_id" in data_json and not st.session_state.conversation_id:
                st.session_state.conversation_id = data_json["conversation_id"]

            choices = data_json.get("choices", [])
            if choices and isinstance(choices, list):
                token = choices[0].get("delta", {}).get("content", "")
                if token:
                    yield token
            elif "content" in data_json:
                yield data_json["content"]
    except requests.RequestException as exc:
        raise APIError("流式连接已中断，请稍后重试") from exc
    finally:
        response.close()


def stream_match_api(
    payload: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any]]]:
    """调用岗位匹配 SSE 流式接口，实时产出 (event_type, payload_dict)。"""

    response = _open_stream(
        f"{API_BASE_URL}/api/match/stream",
        payload,
        timeout_message="岗位匹配流式连接超时",
    )

    try:
        for event_type, data_str in _iter_sse(response, chunk_size=1):
            try:
                yield event_type, json.loads(data_str)
            except json.JSONDecodeError:
                continue
    except requests.RequestException as exc:
        raise APIError("流式连接已中断，请稍后重试") from exc
    finally:
        response.close()


def stream_match_resume_api(
    thread_id: str,
    human_feedback: str | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """调用人工干预与唤醒 SSE 流式接口。"""

    response = _open_stream(
        f"{API_BASE_URL}/api/match/stream/resume",
        {"thread_id": thread_id, "human_feedback": human_feedback or None},
        timeout_message="唤醒工作流流式连接超时",
    )

    try:
        for event_type, data_str in _iter_sse(response, chunk_size=1):
            try:
                yield event_type, json.loads(data_str)
            except json.JSONDecodeError:
                continue
    except requests.RequestException as exc:
        raise APIError("流式连接已中断，请稍后重试") from exc
    finally:
        response.close()
