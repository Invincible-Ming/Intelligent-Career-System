"""Points screens; every amount comes from the authenticated backend wallet."""
from datetime import datetime, timedelta, timezone
from html import escape

import streamlit as st

from career_ui.api import APIError, request_api
from career_ui.render import empty_state, page_header
from career_ui.links import pricing_page_url

LOCAL_TZ = timezone(timedelta(hours=8))
KIND_NAMES = {"initial": "初始赠送", "checkin": "每日签到", "usage": "模型调用"}
STATUS_NAMES = {"pending": "预留中", "settled": "已结算", "released": "已退回预留"}
OPERATIONS = {"/api/chat/completions": "智能对话（含摘要与意图识别）", "knowledge_index": "文件索引",
              "/api/documents/upload": "文档识别", "/api/match": "岗位匹配", "evaluation": "评测"}


def operation_name(value):
    if not value:
        return "—"
    if value in OPERATIONS:
        return OPERATIONS[value]
    for fragment, label in (("/match", "岗位匹配"), ("/interview", "面试准备"),
                             ("/learning-plan", "学习计划"), ("/search", "知识库检索"),
                             ("/documents", "文档处理"), ("/evaluation", "评测")):
        if fragment in value:
            return label
    return "模型处理"


def points(value):
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def _checkin_button(info, *, key, primary=False):
    checked = info["checked_in_today"]
    if st.button("今日已签到" if checked else f"签到领 {info['rules']['daily_checkin_points']} 积分",
                 disabled=checked, key=key, use_container_width=True,
                 type="primary" if primary and not checked else "secondary"):
        result = request_api("POST", "/api/credits/check-in")
        st.toast("签到成功" if result["awarded"] else "今天已经签过到了")
        st.rerun(scope="fragment")


@st.fragment(run_every=10.0)
def render_credit_account():
    try:
        info = request_api("GET", "/api/credits/me")
        with st.container(key="account_wallet"):
            st.html('<div class="account-balance"><span>可用积分</span>'
                    f'<div><strong>{escape(points(info["available"]))}</strong><small> 积分</small></div></div>')
            if info["reserved"]:
                st.caption(f"任务预留 {points(info['reserved'])} 分，结束后结算")
            _checkin_button(info, key="account_checkin", primary=True)
    except APIError as exc:
        st.caption(str(exc))


def _reset_ledger_page():
    st.session_state.credit_entries_offset = 0


@st.dialog("积分记录", width="small")
def show_credit_receipt(row):
    st.caption(datetime.fromisoformat(row["created_at"]).astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M · 北京时间"))
    st.metric(KIND_NAMES.get(row["kind"], "模型用量"), f"{row['points']:+.3f} 分")
    st.write("状态：", STATUS_NAMES.get(row["status"], row["status"]))
    st.write("用途：", operation_name(row.get("operation")))
    if row.get("model"):
        st.write("模型：", row["model"])
        left, right = st.columns(2)
        left.metric("输入 Token", row.get("input_tokens") or 0)
        right.metric("输出 Token", row.get("output_tokens") or 0)
        source = row.get("usage_source") or ""
        st.caption("按服务商返回用量结算" if source.startswith("provider") else "按已有用量或输出保守估算")
    if row["status"] == "pending":
        st.caption(f"本次预留 {points(row['reserved'])} 分，结束后按实际用量结算。")
    if row.get("balance_after") is not None:
        st.caption(f"结算后余额 {points(row['balance_after'])} 分")


@st.fragment
def render_credits_page():
    with st.container(key="credits_page_header"):
        heading, rules_link = st.columns([5, 1], vertical_alignment="center")
        with heading:
            page_header("积分与用量", "每天积累，按实际模型用量使用。", eyebrow="你的账户")
        rules_link.link_button("积分规则", pricing_page_url(), icon=":material/info:", type="tertiary")
    try:
        info = request_api("GET", "/api/credits/me")
        rules = info["rules"]
        with st.container(key="credits_overview", border=True):
            balance, checkin = st.columns([2, 1], gap="large", vertical_alignment="center")
            with balance:
                st.metric("可用积分", points(info["available"]))
                st.caption(f"余额 {points(info['balance'])} · 任务预留 {points(info['reserved'])} 分")
            with checkin:
                _checkin_button(info, key="page_checkin", primary=True)
                st.caption("今天已领取，明天再来" if info["checked_in_today"]
                           else f"每天签到 +{points(rules['daily_checkin_points'])} 分")
        st.caption(f"初始赠送 {points(rules['initial_points'])} 分 · "
                   f"输入与输出合计每千 Token {points(rules['points_per_1000_tokens'])} 分 · 签到每天北京时间 00:00 更新")
        st.session_state.setdefault("credit_entries_offset", 0)
        with st.container(key="credits_ledger"):
            title, filters, refresh = st.columns([4, 1.5, .8], vertical_alignment="center")
            title.subheader("使用记录")
            with filters.popover("筛选", icon=":material/tune:", use_container_width=True):
                if st.session_state.get("credit_filter_kind") not in ("all", "initial", "checkin", "usage"):
                    st.session_state["credit_filter_kind"] = "all"
                if st.session_state.get("credit_filter_status") not in ("all", "settled", "pending", "released"):
                    st.session_state["credit_filter_status"] = "all"
                kind = st.selectbox("记录类型", ["all", "initial", "checkin", "usage"],
                                    format_func=lambda value: "全部类型" if value == "all" else KIND_NAMES[value],
                                    key="credit_filter_kind", on_change=_reset_ledger_page)
                status = st.selectbox("结算状态", ["all", "settled", "pending", "released"],
                                      format_func=lambda value: "全部状态" if value == "all" else STATUS_NAMES[value],
                                      key="credit_filter_status", on_change=_reset_ledger_page)
            if refresh.button("刷新", key="credits_refresh", icon=":material/refresh:", type="tertiary"):
                _reset_ledger_page()
        offset = st.session_state.credit_entries_offset
        params = {"offset": offset, "limit": 20}
        if kind != "all":
            params["kind"] = kind
        if status != "all":
            params["status"] = status
        ledger = request_api("GET", "/api/credits/entries", params=params)
        entries = ledger["items"]
        if entries:
            overview = [{"时间": datetime.fromisoformat(row["created_at"]).astimezone(LOCAL_TZ).strftime("%m-%d %H:%M"),
                         "用途": operation_name(row.get("operation")) if row.get("operation")
                         else KIND_NAMES.get(row["kind"], row["kind"]),
                         "积分变化": f"{row['points']:+.3f}", "状态": STATUS_NAMES.get(row["status"], row["status"])}
                        for row in entries]
            selection = st.dataframe(overview, hide_index=True, width="stretch", on_select="rerun",
                                     selection_mode="single-row", key=f"ledger_{offset}_{kind}_{status}")
            selected = selection.selection.rows
            selected_id = entries[selected[0]]["id"] if selected and 0 <= selected[0] < len(entries) else None
            if selected_id != st.session_state.get("credit_receipt_selection"):
                st.session_state["credit_receipt_selection"] = selected_id
                if selected_id:
                    show_credit_receipt(entries[selected[0]])
            st.caption("选择一条记录，查看模型与用量详情。")
        elif kind != "all" or status != "all":
            st.caption("没有符合当前条件的记录。")
        else:
            empty_state("每一分，都有记录", "签到与模型调用的积分变化会显示在这里。")
        previous, page, next_page = st.columns([1, 4, 1], vertical_alignment="center")
        page.caption(f"第 {offset // 20 + 1} 页")
        if previous.button("上一页", disabled=offset == 0, key="credits_previous", use_container_width=True, type="tertiary"):
            st.session_state.credit_entries_offset = max(0, offset - 20)
            st.rerun(scope="fragment")
        if next_page.button("下一页", disabled=not ledger["has_more"], key="credits_next", use_container_width=True, type="tertiary"):
            st.session_state.credit_entries_offset = offset + 20
            st.rerun(scope="fragment")
        with st.expander("积分规则与结算说明"):
            st.write("输入与输出 Token 合计计费。对话、摘要、意图识别、岗位分析、OCR 与 Embedding 按模型用量结算；"
                     "本地解析、BM25、重排序以及命中缓存的模型步骤不扣积分。管理员也按用量扣分。")
            st.write("调用前预留积分，结束后按用量结算，多退少补且不超过本次预留。优先采用服务商返回用量；"
                     "缺少用量但已有输出时保守估算，记录会注明。失败且没有用量或输出时退回预留。")
            st.write("多步骤任务已完成的模型调用仍会结算；停止生成后，已经产生的用量也会结算。当前不提供充值或现金兑换。")
    except APIError as exc:
        st.error(str(exc))
