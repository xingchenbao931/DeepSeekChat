# -*- coding: utf-8 -*-
"""
DeepSeekChat - DeepSeek 本地对话框架
FastAPI 后端 + llama-cpp-python SSE 流式输出
"""
import os
import sys
import json
import time
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ============ 路径配置 ============
if getattr(sys, 'frozen', False):
    # PyInstaller 打包后的 exe 运行环境
    PROJECT_ROOT = Path(sys.executable).parent
    BUNDLE_DIR = Path(sys._MEIPASS)
else:
    # 源码运行环境
    PROJECT_ROOT = Path(__file__).parent
    BUNDLE_DIR = PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
STATIC_DIR = BUNDLE_DIR / "static"
HISTORY_FILE = PROJECT_ROOT / "history.json"

# ============ 配置参数 ============
CONTEXT_WINDOW = 8192          # 上下文窗口大小
MAX_HISTORY_MESSAGES = 20      # 保留最近的消息数量
N_THREADS = 12                 # CPU 线程数
N_GPU_LAYERS = 0               # GPU 层数 (CPU 模式为 0)
PORT = 7860                    # 服务端口

# ============ 模型路径 ============
# 默认使用 DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M
DEFAULT_MODEL = MODELS_DIR / "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"

app = FastAPI(title="DeepSeekChat")

# ============ 全局状态 ============
llm = None
model_loaded = False
model_path = None


def find_model() -> Path | None:
    """在 models 目录下查找 GGUF 模型文件"""
    if DEFAULT_MODEL.exists():
        return DEFAULT_MODEL
    if MODELS_DIR.exists():
        for f in MODELS_DIR.glob("*.gguf"):
            return f
    return None


def load_model(model_file: Path | None = None):
    """加载 GGUF 模型"""
    global llm, model_loaded, model_path
    from llama_cpp import Llama

    target = model_file or find_model()
    if target is None or not target.exists():
        return False

    try:
        llm = Llama(
            model_path=str(target),
            n_ctx=CONTEXT_WINDOW,
            n_threads=N_THREADS,
            n_gpu_layers=N_GPU_LAYERS,
            use_mmap=True,
            verbose=False,
        )
        model_loaded = True
        model_path = target
        print(f"[DeepSeekChat] 模型加载成功: {target.name}")
        return True
    except Exception as e:
        print(f"[DeepSeekChat] 模型加载失败: {e}")
        llm = None
        model_loaded = False
        return False


# ============ Token 估算与消息裁剪 ============
def _estimate_tokens(messages: list) -> int:
    """粗略估算消息列表的 token 数量 (中文约1字=1.5token, 英文约4字符=1token)"""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        chinese_chars = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
        other_chars = len(content) - chinese_chars
        total += int(chinese_chars * 1.5 + other_chars / 4) + 4
    return total


def _trim_messages(messages: list, keep: int = MAX_HISTORY_MESSAGES) -> list:
    """保留最近的消息,防止上下文溢出"""
    if len(messages) <= keep:
        return messages
    # 保留 system 消息 + 最近的对话
    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]
    trimmed = system_msgs + other_msgs[-keep:]
    return trimmed


def _fit_messages_to_context(messages: list, max_tokens: int = CONTEXT_WINDOW) -> list:
    """动态裁剪消息以适应上下文窗口 (预留 2048 token 给回复)"""
    budget = max_tokens - 2048
    msgs = _trim_messages(messages)
    while _estimate_tokens(msgs) > budget and len(msgs) > 1:
        # 移除最早的非 system 消息
        for i, m in enumerate(msgs):
            if m.get("role") != "system":
                msgs.pop(i)
                break
        else:
            break
    return msgs


# ============ DeepSeek 提示词模板 ============
# DeepSeek-R1-Distill 模型使用 ChatML 格式
def build_messages(user_input: str, history: list) -> list:
    """构建对话消息列表"""
    system_prompt = {
        "role": "system",
        "content": "你是 DeepSeek,一个由深度求索开发的 AI 助手。你乐于助人、诚实友好,请用中文回答用户的问题。"
    }
    messages = [system_prompt]
    # 添加历史对话
    for h in history[-MAX_HISTORY_MESSAGES:]:
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})
    # 添加当前输入
    messages.append({"role": "user", "content": user_input})
    return _fit_messages_to_context(messages)


# ============ API 路由 ============
@app.on_event("startup")
async def startup_event():
    """启动时自动加载模型"""
    global model_loaded
    print("[DeepSeekChat] 正在加载模型...")
    success = load_model()
    if success:
        print(f"[DeepSeekChat] 服务就绪,访问 http://127.0.0.1:{PORT}")
    else:
        print("[DeepSeekChat] 未找到模型,请运行 download_model.py 下载模型")


@app.get("/")
async def index():
    """返回前端页面"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def status():
    """返回模型加载状态"""
    return {
        "loaded": model_loaded,
        "model": model_path.name if model_path else None,
        "context_window": CONTEXT_WINDOW,
        "engine": "llama-cpp-python (CPU)",
    }


@app.post("/api/chat")
async def chat(request: Request):
    """SSE 流式对话接口"""
    global llm, model_loaded

    if not model_loaded or llm is None:
        return JSONResponse(
            {"error": "模型未加载,请先运行 download_model.py 下载模型"},
            status_code=503
        )

    data = await request.json()
    user_input = data.get("message", "").strip()
    history = data.get("history", [])

    if not user_input:
        return JSONResponse({"error": "消息不能为空"}, status_code=400)

    messages = build_messages(user_input, history)

    async def stream_response():
        """SSE 流式生成"""
        try:
            # 使用 llama-cpp 的 create_chat_completion 流式输出
            resp = llm.create_chat_completion(
                messages=messages,
                stream=True,
                max_tokens=2048,
                temperature=0.7,
                top_p=0.9,
            )
            for chunk in resp:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    # SSE 格式发送
                    yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0)  # 让出事件循环
            # 发送结束标记
            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/clear")
async def clear_history():
    """清空对话历史"""
    return {"success": True}


@app.get("/api/models")
async def list_models():
    """列出可用模型"""
    models = []
    if MODELS_DIR.exists():
        for f in MODELS_DIR.glob("*.gguf"):
            size_mb = f.stat().st_size / (1024 * 1024)
            models.append({
                "name": f.name,
                "size": f"{size_mb:.1f} MB",
                "active": str(f) == str(model_path) if model_path else False,
            })
    return {"models": models}


# ============ 静态文件挂载 ============
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============ 主入口 ============
if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  DeepSeekChat - 本地 DeepSeek 对话框架")
    print("=" * 50)
    print(f"  工作目录: {PROJECT_ROOT}")
    print(f"  模型目录: {MODELS_DIR}")
    print(f"  上下文窗口: {CONTEXT_WINDOW} tokens")
    print(f"  CPU 线程: {N_THREADS}")
    print("=" * 50)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
