# ima 知识库接入

通过 `backend/.env` 的 `KNOWLEDGE_PROVIDER=ima` 切换；默认值 `milvus` 保留原有本地混合检索。
切换后重启后端，并刷新前端。不会删除本地文件、数据库记录、Milvus Collection 或 Docker 数据卷。

## 实际能力与边界

本适配器依照腾讯官方 [ima Skill 1.1.10](https://app-dl.ima.qq.com/skills/ima-skills-1.1.10.zip)
内的 `knowledge-base/references/api.md` 实现，使用 POST JSON 和 Client ID / API Key 请求头。
同时兼容实际响应中的 `code/msg` 和 `retcode/errmsg` 格式。

`search_knowledge` 提供知识条目及可选的 `highlight_content`；它不暴露与 Milvus 等价的向量、
相似度或 Dense/BM25 调参接口。适配器使用搜索正文摘要，摘要缺失时尝试读取原文：笔记调用
`get_doc_content`，支持的文件通过受信内容域名下载后交给原有文档沙箱解析。
每次检索最多两页、三份原文，默认总耗时不超过 20 秒。不自动对扫描文档执行付费 OCR。
只有标题、文件夹、无原文权限的条目不会冒充正文进入模型。

因此 ima 适合减少部署和在客户端维护知识资料；需要控制召回、分块、混合检索和消融评测时，
Milvus 模式更适合。本项目的两种模式不是同一种算法，不应直接比较其分数。

## 配置

在 [官方配置页](https://ima.qq.com/agent-interface) 获取凭证，将值分别保存在：

```text
~/.config/ima/client_id
~/.config/ima/api_key
```

目录权限 0700，文件权限 0600。后端 `.env` 配置文件路径，不复制密钥：

```dotenv
KNOWLEDGE_PROVIDER=ima
IMA_BASE_URL=https://ima.qq.com/openapi/wiki/v1
IMA_CLIENT_ID_FILE=/实际路径/.config/ima/client_id
IMA_API_KEY_FILE=/实际路径/.config/ima/api_key
IMA_KNOWLEDGE_BASE_ID=明确供系统共享的知识库ID
IMA_TIMEOUT=8
IMA_SEARCH_TIMEOUT=20
IMA_MAX_SOURCES=3
IMA_MAX_CONTENT_CHARS=6000
```

系统只检索这个指定库。凭证可访问的其他个人库、订阅库不会自动公开给应用用户。
该指定库是**所有已登录用户可检索的共享资料库**，应只放计划提供给系统用户的知识资料。
不同用户的私人简历与 JD 保留在 PostgreSQL / MinIO 中，通过 `owner_id` 过滤，
不会自动上传到 ima，也不会将 ima 凭证交给前端、模型或沙箱。

若后端以后运行在容器中，只读挂载这两个凭证文件，将配置路径改为容器内路径；
不挂载整个用户目录或项目 `.env` 到工具沙箱。API 请求不跟随重定向，也不使用终端遗留代理环境变量。

## 使用

- 在 ima 客户端向配置的共享库添加、更新、删除知识资料；系统知识库页提供管理入口和条目列表。
- 系统原有上传入口仍保存本地文件，ima 模式下只解析和分块，不调用 Embedding 或 Milvus。
  管理员在系统上传的知识资料也保留在本地，不会自动同步云端。
- 本地文件可以按关键词检索；检索范围设为简历或岗位描述时，不请求 ima。
- 智能对话新增 `search_career_knowledge` 工具，用户身份由后端绑定，不接受模型传入 `owner_id`。
- 原有 `POST /api/search/hybrid` 在 ima 模式下执行 ima 与本地关键词检索；
  `/api/search` 和 `/api/search/bm25` 返回 409，避免将 ima 搜索伪装成本地 Dense/BM25。
- 删除本地文档只删除本地记录和原文件；云端资料在 ima 中单独管理。
- 岗位匹配、评分、报告校验、聊天与摘要仍使用百炼。现有匹配流程不会因切换知识库多调用模型。
- Dense/BM25/Hybrid 消融评测仅能在 Milvus 模式启动，ima 模式隐藏新建评测入口；历史报告仍保留。

只读检查，不上传或修改远端资料：

```bash
cd backend
python scripts/check_ima.py --query RAG
```

空库可以正常连接，但没有正文可用于问答。部分订阅库能搜索到条目，却可能没有读取原文的权限；
不能因搜索接口成功就认定原文读取或语义召回质量已经验证。

## Docker 保留项与回退

ima 模式保留业务 PostgreSQL、Redis、业务 MinIO、受限搜索 Broker，以及 Sandbox Runner 和临时任务容器。
原有 Milvus 组可以停用，保留其数据卷方便回退：

```bash
docker compose -f deploy/docker-compose.milvus.yml stop
```

不要删除 `career-minio`：它保存本地简历和岗位文件，与 Milvus 专用 MinIO 不同。

回退时启动 Milvus 组。ima 模式期间新增的本地文件没有向量，需要先重新构建本地索引；
不要只切换配置就把旧索引当成完整索引。提供的回退脚本默认只预检，追加 `--execute` 后
会调用 Embedding，在新的带时间戳 Collection 中按现有分块重建，保留原有索引，且不导入 ima 云端资料。
操作前暂停上传、删除等文档写入，避免重建期间数据发生变化。完成后按脚本输出修改 `.env` 的
`KNOWLEDGE_PROVIDER` 和 `MILVUS_COLLECTION`，再重启后端。

```bash
cd backend
python scripts/reindex_local_documents.py
python scripts/reindex_local_documents.py --execute
```
