"""Streamlit 前端的可复用模块包。

分层：
- config    配置常量
- api       后端客户端（鉴权、超时、SSE）
- session   会话状态与浏览器登录桥接
- endpoints 业务接口薄封装
- render    视图展示组件

app.py 只保留页面编排与样式注入。
"""