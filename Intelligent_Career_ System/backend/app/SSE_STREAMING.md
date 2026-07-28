# SSE 流式对话实现文档

## 📋 概述

你的系统**原本不是 SSE 流式对话**，我已经帮你添加了完整的 SSE 流式支持。

---

## ✅ 已添加的功能

### 1. 流式对话方法 (`bailian.py`)

```python
async def chat_stream(
    self,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
):
    """
    流式调用百炼聊天模型。
    返回异步生成器，逐块产出内容。
    """
    # 启用 stream=True
    # 逐块 yield delta.content
```

**特点**：
- ✅ 使用 OpenAI SDK 的流式接口
- ✅ 异步生成器，实时产出内容
- ✅ 兼容原有的非流式 `chat()` 方法

---

### 2. SSE API 端点 (`chat_api.py`)

```python
@router.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    对话补全接口，支持流式和非流式。
    
    流式返回：Server-Sent Events (SSE)
    非流式返回：JSON
    """
```

**接口格式**：

**请求**：
```json
{
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "temperature": 0.7,
  "stream": true
}
```

**流式响应** (SSE 格式)：
```
data: {"choices":[{"delta":{"content":"你"},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":"好"},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

### 3. 前端示例 (`static/chat_demo.html`)

美观的流式对话界面，包含：

- ✅ 实时打字效果
- ✅ 消息气泡样式
- ✅ 打字指示器动画
- ✅ 自动滚动
- ✅ 对话历史管理
- ✅ 错误处理

---

## 🚀 使用方法

### 1. 启动服务

```bash
python -m uvicorn app.main:app --reload
```

### 2. 访问前端演示

打开浏览器访问：
```
http://localhost:8000/static/chat_demo.html
```

### 3. 使用 API

#### Python 客户端示例

```python
import httpx

async def stream_chat():
    url = "http://localhost:8000/api/chat/completions"
    
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            url,
            json={
                "messages": [
                    {"role": "user", "content": "介绍一下 Python"}
                ],
                "temperature": 0.7,
                "stream": True
            },
            timeout=60.0
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    
                    import json
                    chunk = json.loads(data)
                    if chunk["choices"][0]["delta"].get("content"):
                        print(chunk["choices"][0]["delta"]["content"], end="", flush=True)

import asyncio
asyncio.run(stream_chat())
```

#### JavaScript/Fetch 示例

```javascript
async function streamChat(message) {
    const response = await fetch('http://localhost:8000/api/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            messages: [
                { role: 'user', content: message }
            ],
            temperature: 0.7,
            stream: true
        })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') return;

                const parsed = JSON.parse(data);
                const content = parsed.choices[0].delta.content;
                if (content) {
                    console.log(content); // 实时打印
                }
            }
        }
    }
}

streamChat('你好');
```

#### cURL 测试

```bash
curl -N -X POST http://localhost:8000/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.7,
    "stream": true
  }'
```

---

## 📊 对比：流式 vs 非流式

| 特性 | 非流式（原有） | 流式（新增） |
|------|--------------|-------------|
| **响应方式** | 一次性返回完整答案 | 逐字/逐块返回 |
| **用户体验** | 需要等待完整生成 | 实时看到生成过程 |
| **接口类型** | JSON | Server-Sent Events (SSE) |
| **适用场景** | 结构化输出、评测 | 对话交互、实时反馈 |
| **实现方式** | `chat()` 方法 | `chat_stream()` 方法 |

---

## 🔧 配置调整

### 启用静态文件服务

在 `main.py` 中添加：

```python
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="app/static"), name="static")
```

### CORS 设置（如果前端单独部署）

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🎯 应用场景

### ✅ 1. 实时对话

```python
# 在你的 workflow.py 中使用流式
async def stream_match_report(resume_text: str, jd_text: str):
    """流式生成匹配报告"""
    messages = [
        {"role": "system", "content": "你是求职分析专家"},
        {"role": "user", "content": f"分析简历：{resume_text}\n岗位：{jd_text}"}
    ]
    
    async for chunk in bailian_service.chat_stream(messages):
        yield chunk  # 实时返回分析结果
```

### ✅ 2. 面试问题生成

```python
async def stream_interview_questions(match_report: dict):
    """流式生成面试问题"""
    prompt = f"根据匹配报告生成面试问题：{match_report}"
    
    messages = [{"role": "user", "content": prompt}]
    
    async for chunk in bailian_service.chat_stream(messages):
        yield chunk
```

### ✅ 3. 学习计划生成

```python
async def stream_learning_plan(gaps: list[str]):
    """流式生成学习计划"""
    prompt = f"为以下技能缺口制定学习计划：{gaps}"
    
    messages = [{"role": "user", "content": prompt}]
    
    async for chunk in bailian_service.chat_stream(messages):
        yield chunk
```

---

## 🐛 常见问题

### Q1: SSE 连接中断

**原因**: 网络超时或代理服务器限制

**解决**:
```python
# 增加超时时间
response = await client.stream(..., timeout=120.0)

# 或使用心跳保活
async def stream_with_heartbeat():
    async for chunk in chat_stream(...):
        yield chunk
        await asyncio.sleep(0)  # 保持连接活跃
```

### Q2: 前端无法接收流式数据

**原因**: CORS 限制

**解决**: 在 FastAPI 中添加 CORS 中间件

### Q3: 流式输出不完整

**原因**: 缓冲区问题

**解决**:
```python
# 确保立即刷新
yield f"data: {json.dumps(data)}\n\n"
await asyncio.sleep(0)  # 强制刷新
```

---

## 📈 性能对比

### 非流式（原有）

- **首字延迟**: 5-10 秒（等待完整生成）
- **用户感知**: 长时间等待
- **适合**: 结构化输出、批处理

### 流式（新增）

- **首字延迟**: 0.5-1 秒（立即开始）
- **用户感知**: 实时反馈
- **适合**: 对话交互、实时展示

---

## 🎓 最佳实践

1. **对话场景用流式**: 用户交互体验更好
2. **结构化输出用非流式**: 便于解析和验证
3. **长文本用流式**: 避免用户长时间等待
4. **短文本可非流式**: 减少网络开销
5. **评测系统用非流式**: 便于指标计算

---

## 📚 参考资料

- [Server-Sent Events (SSE) 规范](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [OpenAI Streaming API](https://platform.openai.com/docs/api-reference/streaming)
- [FastAPI StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)

---

**总结**: 你的系统现在**同时支持流式和非流式对话**了！🎉
