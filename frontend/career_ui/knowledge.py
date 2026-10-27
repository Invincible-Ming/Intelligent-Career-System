"""Knowledge page panels rerun independently; forms batch changes until submission."""
from __future__ import annotations

import time
import mimetypes
from pathlib import Path

import streamlit as st

from career_ui.api import APIError, security_options
from career_ui.config import DOCUMENT_TYPE_NAMES, SEARCH_MODE_NAMES
from career_ui.documents import document_name_marker, document_upload_time
from career_ui.endpoints import (
    delete_document,
    download_document,
    get_document_details,
    get_documents,
    get_knowledge_items,
    get_knowledge_status,
    search_documents,
    rename_document,
    retry_document,
    upload_document,
)
from career_ui.render import empty_state, page_header, show_document_preview, show_search_results
from career_ui.retrieval import knowledge_strategy, uses_ima
from career_ui.retrieval import render_index_progress, render_provider_choice


def invalidate_document_list():
    st.session_state["knowledge_documents_revision"] = st.session_state.get("knowledge_documents_revision", 0) + 1
    st.session_state.pop("knowledge_documents_cache", None)


def close_knowledge_dialog():
    st.session_state.pop("knowledge_dialog", None)
    st.session_state.pop("knowledge_delete_target", None)
    st.session_state.pop("knowledge_delete_targets", None)
    st.session_state.pop("knowledge_document_target", None)
    st.session_state.pop("knowledge_rename_name", None)


def document_download_button(document, *, key):
    # Capture this session's identity, never access session_state in the deferred
    # worker and never prefetch all originals merely to display a file list.
    token = st.session_state.get("access_token") or ""
    identifier = str(document["id"])
    st.download_button("下载原文件", data=lambda: download_document(identifier, token=token),
                       file_name=document["filename"],
                       mime=mimetypes.guess_type(document["filename"])[0] or "application/octet-stream",
                       key=key, icon=":material/download:", type="tertiary", on_click="ignore",
                       use_container_width=True)


def document_action_complete(message):
    invalidate_document_list()
    st.session_state["knowledge_document_notice"] = message
    close_knowledge_dialog()
    st.rerun(scope="app")


def file_size_label(size):
    if size is None:
        return "未知"
    return f"{size / 1024 / 1024:.2f} MB" if size >= 1024 * 1024 else f"{size / 1024:.1f} KB"


@st.dialog("文件详情", width="small", on_dismiss=close_knowledge_dialog)
def document_details_dialog(document):
    try:
        detail = get_document_details(document["id"])
    except APIError as exc:
        st.error(str(exc))
        return
    st.text(detail["filename"])
    fields = [
        ("文档类型", DOCUMENT_TYPE_NAMES.get(detail.get("document_type"), "文件")),
        ("文件大小", file_size_label(detail.get("size_bytes"))),
        ("文件格式", Path(detail["filename"]).suffix.lstrip(".").upper()),
        ("上传时间", document_upload_time(detail)),
        ("处理状态", {"ready": "可用", "failed": "处理失败", "processing": "处理中"}.get(detail.get("status"), "未知")),
    ]
    with st.container(key="knowledge_document_metadata"):
        for title, content in fields:
            label, value = st.columns([1, 2], vertical_alignment="center")
            label.caption(title)
            value.text(content)
    if detail.get("error_message"):
        st.warning(detail["error_message"])
    elif detail.get("status") == "ready" and not detail.get("preview_available"):
        st.caption("这份历史文件未保存可预览的正文，仍可下载原文件。")
    document_download_button(detail, key="knowledge_detail_download")


@st.dialog("重命名文件", width="small", on_dismiss=close_knowledge_dialog)
def rename_document_dialog(document):
    suffix = Path(document["filename"]).suffix
    st.session_state.setdefault("knowledge_rename_name", document["filename"][:-len(suffix)] if suffix else document["filename"])
    with st.form("knowledge_rename_form", border=False):
        name = st.text_input("文件名称", key="knowledge_rename_name", max_chars=200 - len(suffix))
        st.caption(f"扩展名 {suffix} 保持不变；重命名不会重新解析或扣除积分。")
        submitted = st.form_submit_button("保存名称", type="primary", use_container_width=True)
    if submitted:
        if not name.strip():
            st.error("请输入文件名称")
            return
        try:
            result = rename_document(document["id"], name.strip() + suffix)
        except APIError as exc:
            st.error(str(exc))
        else:
            document_action_complete(f"已重命名为：{result['filename']}")


@st.dialog("重试处理", width="small", on_dismiss=close_knowledge_dialog)
def retry_document_dialog(document):
    st.text(document["filename"])
    if document.get("error_message"):
        st.warning(document["error_message"])
    st.caption("将使用原文件重新解析，文件编号不变。扫描页可能调用 OCR 并按实际模型用量扣除积分；原文件始终保留。")
    if st.button("开始重试", key="knowledge_confirm_retry", type="primary", use_container_width=True):
        try:
            with st.spinner("正在重新处理文件，请稍候…"):
                retry_document(document["id"])
        except APIError as exc:
            invalidate_document_list()
            st.error(str(exc))
        else:
            document_action_complete("文件重新处理完成")


def show_document_action_dialog(action, document):
    st.session_state["knowledge_dialog"] = action
    st.session_state["knowledge_document_target"] = document
    {"details": document_details_dialog, "rename": rename_document_dialog,
     "retry": retry_document_dialog}[action](document)


def set_document_selection(identifiers, selected):
    for identifier in identifiers:
        st.session_state[f"knowledge_selected_file_{identifier}"] = selected


def toggle_document_management():
    st.session_state["knowledge_bulk_mode"] = not st.session_state.get("knowledge_bulk_mode", False)


@st.fragment
def render_ima_library():
    description, action = st.columns([3, 1], vertical_alignment="center")
    description.markdown("**共享知识资料**")
    action.link_button("打开 ima", "https://ima.qq.com/", use_container_width=True)
    st.caption("共享资料在 ima 中维护；本地上传文件仅当前账号可用，不会自动上传。")
    try:
        library = get_knowledge_status()
        st.markdown(f"**已连接：{library.get('name', 'ima 知识库')}**")
        with st.expander("查看 ima 中的资料"):
            st.session_state.setdefault("ima_items_cursor", "")
            if st.button("刷新资料列表", key="ima_refresh"):
                st.session_state.ima_items_cursor = ""
            page = get_knowledge_items(st.session_state.ima_items_cursor)
            for item in page.get("items", []):
                st.text(item.get("title", "未命名资料"))
            if not page.get("items"):
                st.info("暂无资料，请先在 ima 中向该知识库添加内容。")
            if not page.get("is_end", True) and page.get("next_cursor"):
                if st.button("下一页", key="ima_next_page"):
                    st.session_state.ima_items_cursor = page["next_cursor"]
                    st.rerun(scope="fragment")
    except APIError as exc:
        st.warning(str(exc))


def render_upload(*, ima_mode, queued_indexing, is_admin):
    notice = st.session_state.pop("upload_notice", None)
    if notice:
        (st.info if notice["kind"] == "dedup" else st.success)(notice["text"])
    types = ["resume", "job_description", "knowledge"] if is_admin else ["resume", "job_description"]
    # File selection and document-type changes stay in the browser until the form is submitted.
    with st.form("knowledge_upload_form", border=False):
        document_type = st.selectbox(
            "文档类型", options=types, format_func=DOCUMENT_TYPE_NAMES.get, key="knowledge_upload_type",
            help=None if is_admin else "知识资料由管理员维护，如需上传请联系管理员",
        )
        file = st.file_uploader(
            "选择文档文件", type=["pdf", "docx", "xlsx", "txt", "md"],
            max_upload_size=security_options()["max_upload_size_mb"],
            key=f"knowledge_uploader_{st.session_state.get('uploader_nonce', 0)}",
            help="支持 PDF、DOCX、XLSX、TXT 和 Markdown（.md）格式",
        )
        submitted = st.form_submit_button(
            "上传并解析" if ima_mode or queued_indexing else "上传并向量化入库",
            type="primary", use_container_width=True,
        )
    if not submitted:
        return
    if file is None:
        st.info("请先选择要上传的文件")
        return
    try:
        with st.spinner("正在安全解析并保存文件..." if ima_mode or queued_indexing
                        else "正在解析、切片、向量化并建立索引..."):
            result = upload_document(file, document_type)
        document = result.get("document") or {}
        selection_key = {"resume": "match_resume_document_id", "job_description": "match_job_document_id"}.get(
            document_type,
        )
        if selection_key and document.get("id"):
            st.session_state[selection_key] = str(document["id"])
        deduplicated = result.get("deduplicated")
        st.session_state["upload_notice"] = {
            "kind": "dedup" if deduplicated else "new",
            "text": (f"这份内容已上传过：{document.get('filename', file.name)}，本次未重复处理。"
                     if deduplicated else f"文件已保存：{file.name}"),
        }
        st.session_state["uploader_nonce"] = st.session_state.get("uploader_nonce", 0) + 1
        invalidate_document_list()
        close_knowledge_dialog()
        st.rerun(scope="app")
    except APIError as exc:
        st.error(str(exc))


@st.dialog("上传文件", width="small", on_dismiss=close_knowledge_dialog)
def upload_dialog(*, ima_mode, queued_indexing, is_admin):
    st.caption("上传后安全解析并保存到你的资料空间。")
    render_upload(ima_mode=ima_mode, queued_indexing=queued_indexing, is_admin=is_admin)


@st.dialog("检索设置", width="small", on_dismiss=close_knowledge_dialog)
def retrieval_dialog(user, *, queued_indexing):
    st.caption(knowledge_strategy(user))
    render_provider_choice(user)
    if queued_indexing and not uses_ima(user):
        render_index_progress()
    if uses_ima(user):
        render_ima_library()


@st.dialog("删除文件", width="small", on_dismiss=close_knowledge_dialog)
def confirm_delete_document(document):
    documents = document if isinstance(document, list) else [document]
    st.write(f"确定删除以下 {len(documents)} 份文件？" if len(documents) > 1 else "确定删除这份文件？")
    for item in documents:
        st.text(item.get("filename", "未命名文档"))
    st.caption("文件及其检索索引将被删除，此操作无法撤销。已有匹配报告保留。")
    confirm, cancel = st.columns(2)
    if confirm.button("确认删除", key="confirm_document_delete", type="primary", use_container_width=True):
        failed, removed = [], []
        for item in documents:
            try:
                delete_document(item["id"])
                removed.append(item["id"])
            except APIError as exc:
                failed.append(item)
                st.error(f"{item['filename']}：{exc}")
        if removed:
            invalidate_document_list()
            st.session_state["knowledge_selection_reset"] = removed
            st.session_state["knowledge_document_notice"] = f"已删除 {len(removed)} 份文件"
        if not failed:
            close_knowledge_dialog()
            st.rerun(scope="app")
        elif removed:
            st.session_state["knowledge_delete_targets"] = failed
            st.session_state["knowledge_delete_target"] = None
            st.rerun(scope="app")
    if cancel.button("取消", key="cancel_document_delete", use_container_width=True):
        close_knowledge_dialog()
        st.rerun(scope="app")


@st.fragment
def render_search(*, ima_mode, queued_indexing):
    user = st.session_state.get("auth_user") or {}
    st.caption("搜索共享知识与自己的文件" if ima_mode else "搜索自己的资料")
    result_scope = (user.get("id", user.get("username")), ima_mode, queued_indexing)
    if st.session_state.get("knowledge_search_result_scope") != result_scope:
        st.session_state.pop("search_results", None)
        st.session_state.pop("search_mode", None)
        st.session_state["knowledge_search_result_scope"] = result_scope
    query_column, action_column, filters_column = st.columns([6, 1.2, 1.2], vertical_alignment="bottom")
    with query_column:
        query = st.text_input("检索问题", placeholder="向资料提问，例如：我的项目经历能体现哪些架构能力？",
                              max_chars=security_options()["search_max_input_chars"],
                              key="knowledge_search_query", label_visibility="collapsed")
    with action_column:
        submitted = st.button("执行检索", key="knowledge_search_submit", type="primary", use_container_width=True)
    with filters_column.popover("筛选", icon=":material/tune:"):
        mode = "ima" if ima_mode else "hybrid"
        if not queued_indexing and not ima_mode:
            mode = st.selectbox("检索策略", options=["hybrid", "dense", "bm25"],
                                format_func=SEARCH_MODE_NAMES.get, key="knowledge_search_strategy")
        scope_column, count_column = st.columns(2)
        with scope_column:
            if st.session_state.get("knowledge_search_document_type") not in ("all", "resume", "job_description", "knowledge"):
                st.session_state["knowledge_search_document_type"] = "all"
            scope = st.selectbox("检索范围", options=["all", "resume", "job_description", "knowledge"],
                                         format_func=lambda value: "全部资料" if value == "all"
                                         else DOCUMENT_TYPE_NAMES[value], key="knowledge_search_document_type")
            document_type = None if scope == "all" else scope
        with count_column:
            top_k = st.slider("结果数量", min_value=1, max_value=10,
                              value=None if "knowledge_search_top_k" in st.session_state else 5,
                              key="knowledge_search_top_k")
        st.caption("ima 可查询共享资料，本地文件仅当前账号可用。" if ima_mode
                   else "仅检索当前账号的本地文件。")
    if submitted:
        if not query.strip():
            st.info("请输入检索问题")
        else:
            try:
                with st.spinner(f"正在执行 {SEARCH_MODE_NAMES[mode]}..."):
                    results = search_documents(query=query, document_type=document_type, top_k=top_k, search_mode=mode)
                st.session_state["search_results"] = results
                st.session_state["search_mode"] = mode
            except APIError as exc:
                st.error(str(exc))
    results = st.session_state.get("search_results", [])
    if results:
        st.divider()
        st.subheader("检索结果")
        show_search_results(results, st.session_state.get("search_mode", "hybrid"))
    elif "search_results" in st.session_state:
        st.info("没有找到可读取的相关正文。可尝试关键词，并确认 ima 中已有资料且原文可访问。"
                if ima_mode else "没有检索到相关内容。")


@st.fragment
def render_document_list(*, ima_mode):
    user = st.session_state.get("auth_user") or {}
    owner = user.get("id", user.get("username"))
    if st.session_state.get("knowledge_selection_owner") != owner:
        for key in list(st.session_state):
            if key.startswith("knowledge_selected_file_"):
                del st.session_state[key]
        st.session_state["knowledge_selection_owner"] = owner
        st.session_state["knowledge_bulk_mode"] = False
    title, manage, action = st.columns([4, 1.2, .9], vertical_alignment="center")
    title.subheader("我的文件")
    manage.button("完成管理" if st.session_state.get("knowledge_bulk_mode") else "批量管理",
                  key="knowledge_manage_documents", type="tertiary", on_click=toggle_document_management)
    refresh = action.button("刷新", key="knowledge_refresh_documents", icon=":material/refresh:", type="tertiary")
    notice = st.session_state.pop("knowledge_document_notice", None)
    if notice:
        st.success(notice)
    # Session-only, owner-bound cache. Mutations invalidate it; explicit refresh
    # and cache expiry catch external changes without collapsing an open menu.
    for identifier in st.session_state.pop("knowledge_selection_reset", []):
        st.session_state.pop(f"knowledge_selected_file_{identifier}", None)
    revision = st.session_state.get("knowledge_documents_revision", 0)
    cache = st.session_state.get("knowledge_documents_cache")
    needs_load = (refresh or cache is None or cache["owner"] != owner or cache["revision"] != revision
                  or time.monotonic() - cache["loaded_at"] >= 30)
    try:
        if needs_load:
            documents = get_documents()
            cache = {"owner": owner, "revision": revision, "loaded_at": time.monotonic(), "items": documents}
            st.session_state["knowledge_documents_cache"] = cache
        documents = cache["items"]
    except APIError as exc:
        st.error(str(exc))
        return
    if not documents:
        empty_state("你的文件会显示在这里", "上传简历、岗位描述或知识资料后，可以随时查看与管理。")
        return
    filter_column, type_column = st.columns([3, 2], vertical_alignment="center")
    filename_filter = filter_column.text_input("查找文件", placeholder="按文件名查找", key="knowledge_file_filter",
                                               label_visibility="collapsed")
    if st.session_state.get("knowledge_file_type") is None:
        st.session_state["knowledge_file_type"] = "all"
    kind = type_column.selectbox("文件类型", ["all", "resume", "job_description", "knowledge"],
                                format_func=lambda value: "全部类型" if value == "all" else DOCUMENT_TYPE_NAMES[value],
                                key="knowledge_file_type", label_visibility="collapsed")
    documents = [doc for doc in documents if (kind == "all" or doc.get("document_type") == kind)
                 and filename_filter.casefold() in doc.get("filename", "").casefold()]
    st.caption(f"{len(documents)} 份文件 · 最近上传优先")
    if not documents:
        st.caption("没有符合条件的文件，试试其他名称或类型。")
    selectable = [doc["id"] for doc in documents if doc.get("status") != "processing"]
    bulk_mode = st.session_state.get("knowledge_bulk_mode", False)
    # A filter change never leaves an invisible file selected for deletion.
    for key in list(st.session_state):
        if key.startswith("knowledge_selected_file_") and key.removeprefix("knowledge_selected_file_") not in selectable:
            st.session_state[key] = False
    if bulk_mode and documents:
        select_all, clear, remove = st.columns([1, 1, 3], vertical_alignment="center")
        select_all.button("全选", key="knowledge_select_all", type="tertiary",
                          on_click=set_document_selection, args=(selectable, True))
        clear.button("清空", key="knowledge_clear_selection", type="tertiary",
                     on_click=set_document_selection, args=(selectable, False))
        selected = [doc for doc in documents if st.session_state.get(f"knowledge_selected_file_{doc['id']}")]
        if remove.button(f"删除所选（{len(selected)}）", key="knowledge_batch_delete", disabled=not selected):
            st.session_state["knowledge_dialog"] = "delete"
            st.session_state["knowledge_delete_targets"] = selected
            confirm_delete_document(selected)
            return True
    for document in documents:
        filename, document_id = document.get("filename", "未命名文档"), document.get("id", "")
        if bulk_mode:
            choice, first, third, fourth = st.columns([.35, 6, 1.4, .55], vertical_alignment="center")
            choice.checkbox(f"选择 {filename}", key=f"knowledge_selected_file_{document_id}",
                            disabled=document.get("status") == "processing", label_visibility="collapsed")
        else:
            first, third, fourth = st.columns([6, 1.4, .55], vertical_alignment="center")
        if first.button(filename, key=f"preview-document-{document_id}", icon=":material/description:",
                        disabled=document.get("status") != "ready", use_container_width=True,
                        type="tertiary", help="查看文件正文"):
            show_document_preview(document)
        marker = document_name_marker(document)
        first.caption(f"{DOCUMENT_TYPE_NAMES.get(document.get('document_type'), '文件')} · {document_upload_time(document)}"
                      + (f" · {marker}" if marker else ""))
        status = document.get("status", "unknown")
        if status == "ready":
            third.html('<span class="document-status">可用</span>')
        elif status == "failed":
            third.html('<span class="document-status failed">处理失败</span>')
        else:
            third.html('<span class="document-status pending">处理中</span>')
        with fourth.popover("", icon=":material/more_horiz:", help=f"管理 {filename}"):
            document_download_button(document, key=f"download-document-{document_id}")
            for action, label, icon in (("rename", "重命名", "edit"), ("details", "查看详情", "info")):
                if st.button(label, key=f"{action}-document-{document_id}", icon=f":material/{icon}:",
                             disabled=action == "rename" and status == "processing",
                             type="tertiary", use_container_width=True):
                    show_document_action_dialog(action, document)
                    return True
            if status == "failed" and st.button("重试处理", key=f"retry-document-{document_id}",
                                               icon=":material/refresh:", type="tertiary", use_container_width=True):
                show_document_action_dialog("retry", document)
                return True
            st.divider()
            if st.button("删除文件", key=f"delete-document-{document_id}", type="tertiary",
                         icon=":material/delete:", disabled=status == "processing", use_container_width=True):
                st.session_state["knowledge_dialog"] = "delete"
                st.session_state["knowledge_delete_target"] = document
                confirm_delete_document(document)
                return True


def render_knowledge_page(user, *, queued_indexing):
    page_header("你的资料", "把资料放在一起，让每一次提问都有依据。", eyebrow="知识库")
    is_admin = user.get("is_admin", False)
    ima_mode = uses_ima(user)
    with st.container(key="knowledge_toolbar"):
        description, upload, settings = st.columns([4, 1.25, .7], vertical_alignment="center")
        description.caption("简历、岗位与学习资料")
        if upload.button("上传文件", key="knowledge_open_upload", icon=":material/add:", use_container_width=True):
            st.session_state["knowledge_dialog"] = "upload"
        if settings.button("设置", key="knowledge_open_settings", icon=":material/tune:", type="tertiary"):
            st.session_state["knowledge_dialog"] = "settings"
    notice = st.session_state.pop("upload_notice", None)
    if notice:
        st.success(notice["text"])
    with st.container(key="knowledge_search_panel"):
        render_search(ima_mode=ima_mode, queued_indexing=queued_indexing)
    with st.container(key="knowledge_files_panel"):
        modal_opened = render_document_list(ima_mode=ima_mode)
    dialog = st.session_state.get("knowledge_dialog")
    if dialog == "upload":
        upload_dialog(ima_mode=ima_mode, queued_indexing=queued_indexing, is_admin=is_admin)
    elif dialog == "settings":
        retrieval_dialog(user, queued_indexing=queued_indexing)
    elif dialog == "delete" and not modal_opened:
        targets = st.session_state.get("knowledge_delete_targets") or st.session_state.get("knowledge_delete_target")
        if targets:
            confirm_delete_document(targets)
    elif dialog in ("details", "rename", "retry") and not modal_opened and st.session_state.get("knowledge_document_target"):
        show_document_action_dialog(dialog, st.session_state["knowledge_document_target"])
