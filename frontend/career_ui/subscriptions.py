"""Subscription controls backed by authenticated server entitlements."""
from datetime import UTC, datetime, timedelta, timezone

import streamlit as st

from career_ui.api import APIError, request_api

PLAN_NAMES = {"free": "免费版", "starter": "入门版", "pro": "专业版", "enterprise": "企业版"}
LOCAL_TZ = timezone(timedelta(hours=8))


def render_subscription_summary(user):
    subscription = user.get("subscription") or {}
    st.caption(f"当前方案：{subscription.get('plan_name', '免费版')}")
    expires = subscription.get("expires_at")
    if expires:
        when = datetime.fromisoformat(expires).astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
        st.caption(f"{'已到期' if subscription.get('expired') else '到期时间'}：{when}")


def render_subscription_admin():
    # Keep the disclosure outside the fragment so changing accounts cannot collapse/recreate it.
    with st.expander("订阅管理"):
        st.caption("管理员开通后生效，到期自动恢复免费版。当前尚未接入在线支付。")
        _render_subscription_editor()


@st.fragment
def _render_subscription_editor():
    query_col, refresh_col = st.columns([4, 1], vertical_alignment="bottom")
    with query_col:
        query = st.text_input("查找账号", placeholder="输入用户名", max_chars=32, key="subscription_user_query").strip()
    with refresh_col:
        refresh = st.button("刷新账号", key="subscription_refresh_users")
    actor = st.session_state.get("auth_user") or {}
    owner = actor.get("id", actor.get("username"))
    cache = st.session_state.get("subscription_users_cache")
    try:
        # Selecting another cached account is local; refresh only on an explicit request, search, or grant.
        if refresh or cache is None or cache["owner"] != owner or cache["query"] != query:
            users = request_api("GET", "/api/subscriptions/users", params={"query": query})
            cache = {"owner": owner, "query": query, "items": users}
            st.session_state["subscription_users_cache"] = cache
        users = cache["items"]
    except APIError as exc:
        st.error(str(exc))
        return
    if not users:
        st.info("没有找到账号")
        return
    by_id = {item["id"]: item for item in users}
    selected = st.selectbox("账号", options=list(by_id),
                            format_func=lambda value: by_id[value]["username"], key="subscription_target")
    target = by_id[selected]
    subscription = target["subscription"]
    st.caption(f"正在管理：{target['username']} · 当前：{subscription['plan_name']}")
    defaults = st.session_state.get("subscription_expiry_defaults")
    fingerprint = (owner, selected, subscription["plan"], subscription.get("expires_at"))
    if defaults is None or defaults["fingerprint"] != fingerprint:
        expiry = (datetime.fromisoformat(subscription["expires_at"]).astimezone(LOCAL_TZ)
                  if subscription.get("expires_at") else datetime.now(LOCAL_TZ) + timedelta(days=30))
        defaults = {"fingerprint": fingerprint, "expiry": expiry}
        st.session_state["subscription_expiry_defaults"] = defaults
    default_expiry = defaults["expiry"]
    with st.form(f"subscription_grant_{selected}"):
        plan = st.selectbox("方案", options=list(PLAN_NAMES), format_func=PLAN_NAMES.get,
                            index=list(PLAN_NAMES).index(subscription["plan"]))
        expires_date = st.date_input("到期日期（免费版不使用此项）", value=default_expiry.date())
        expires_time = st.time_input("到期时间（北京时间）", value=default_expiry.time().replace(tzinfo=None))
        submitted = st.form_submit_button("保存方案", type="primary")
    if submitted:
        expires = datetime.combine(expires_date, expires_time, tzinfo=LOCAL_TZ).astimezone(UTC)
        try:
            request_api("PUT", f"/api/subscriptions/users/{selected}", json={
                "plan": plan, "expires_at": expires.isoformat() if plan != "free" else None,
            })
            st.session_state.pop("subscription_users_cache", None)
            st.session_state.pop("subscription_expiry_defaults", None)
            st.toast("订阅方案已更新")
            # Only a change to the current account needs to rebuild navigation and plan badges.
            st.rerun(scope="app" if selected == actor.get("id") else "fragment")
        except APIError as exc:
            st.error(str(exc))


@st.fragment(run_every=5.0)
def render_index_progress():
    try:
        info = request_api("GET", "/api/knowledge/status")
        if info.get("provider") != "milvus":
            st.rerun()  # The subscription may have expired while this page was open.
        counts = info.get("index") or {}
        total = counts.get("total", 0)
        if total and counts.get("ready", 0) < total:
            st.progress(counts["ready"] / total, text=f"正在准备文件检索：{counts['ready']} / {total}")
            if counts.get("failed"):
                st.warning(f"{counts['failed']} 份文件准备失败，请重试或联系管理员。")
            refresh_col, retry_col = st.columns(2)
            with refresh_col:
                if st.button("刷新进度", key="refresh_index_progress"):
                    st.rerun(scope="fragment")
            with retry_col:
                if st.button("重新准备", key="retry_index_progress"):
                    request_api("POST", "/api/knowledge/index/retry")
                    st.rerun(scope="fragment")
        elif total:
            st.caption(f"{total} 份文件已可用于增强检索")
    except APIError as exc:
        st.warning(str(exc))
