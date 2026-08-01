# -*- coding: utf-8 -*-
"""
DeepSeekChat - DeepSeek 本地对话框架
FastAPI 后端 + llama-cpp-python SSE 流式输出 (异步线程架构)
"""
import os
import sys
import json
import time
import asyncio
import threading
import queue as _queue
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ============ 线程安全的队列标记 ============
_SENTINEL_DONE = "__DONE__"
_SENTINEL_ERROR = "__ERROR__"

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
# 默认使用 Qwen2.5-3B-Instruct Q4_K_M (中文能力强, 与 DeepSeek-R1-Distill-Qwen 架构兼容)
# find_model() 会自动扫描 models 目录下的任意 .gguf 文件, 因此也可以手动放其他兼容模型
DEFAULT_MODEL = MODELS_DIR / "qwen2.5-3b-it-Q4_K_M-LOT.gguf"

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
    """SSE 流式对话接口 (后台线程队列架构, 防止事件循环阻塞)"""
    global llm, model_loaded

    data = await request.json()

    # ============ 双格式参数兼容 ============
    # 格式1: DeepSeekChat 风格 {message, history}
    # 格式2: QwenChat / OpenAI 风格 {messages[], temperature, max_tokens, system_prompt}
    if "messages" in data:
        raw_messages = data.get("messages", [])
        temperature = float(data.get("temperature", 0.7))
        max_tokens = int(data.get("max_tokens", 2048))
        system_prompt_override = data.get("system_prompt", None)
        messages = []
        if system_prompt_override:
            messages.append({"role": "system", "content": system_prompt_override})
        for m in raw_messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role in ("system", "user", "assistant"):
                messages.append({"role": role, "content": content})
        messages = _fit_messages_to_context(messages)
        user_input = messages[-1].get("content", "") if messages else ""
    else:
        user_input = data.get("message", "").strip()
        history = data.get("history", [])
        temperature = 0.7
        max_tokens = 2048
        if not user_input:
            return JSONResponse({"error": "消息不能为空"}, status_code=400)
        messages = build_messages(user_input, history)

    # ============ 技能指令拦截 (>> 前缀) ============
    if user_input.startswith(">>"):
        skill_result = _execute_skill(user_input)
        if skill_result is not None:
            # 以 SSE 流式返回技能执行结果
            async def skill_stream():
                for line in skill_result.split("\n"):
                    yield f"data: {json.dumps({'content': line + chr(10), 'delta': line + chr(10)}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.005)
                yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
            return StreamingResponse(
                skill_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                         "Keep-Alive": "timeout=300", "X-Accel-Buffering": "no"},
            )

    if not model_loaded or llm is None:
        return JSONResponse(
            {"error": "模型未加载,请先运行 download_model.py 下载模型"},
            status_code=503
        )

    # ============ 后台推理线程 + Queue 非阻塞架构 ============
    q: _queue.Queue = _queue.Queue()

    def _worker():
        try:
            resp = llm.create_chat_completion(
                messages=messages,
                stream=True,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            for chunk in resp:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    q.put(content)
            q.put(_SENTINEL_DONE)
        except Exception as e:
            q.put(_SENTINEL_ERROR)
            q.put(str(e))

    threading.Thread(target=_worker, daemon=True).start()

    async def stream_response():
        while True:
            try:
                item = q.get_nowait()
            except _queue.Empty:
                await asyncio.sleep(0.01)
                continue
            if item == _SENTINEL_DONE:
                yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
                break
            if item == _SENTINEL_ERROR:
                try:
                    err_msg = q.get_nowait()
                except _queue.Empty:
                    err_msg = "未知推理错误"
                yield f"data: {json.dumps({'error': err_msg}, ensure_ascii=False)}\n\n"
                break
            # 同时发送 content 和 delta, 前端兼容两种格式
            yield f"data: {json.dumps({'content': item, 'delta': item}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Keep-Alive": "timeout=300",
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


# =====================================================================
# 渗透测试工具 API
# =====================================================================
from pentest import DISCLAIMER
from pentest.utils import (
    md5, b64e, b64d, url_encode, url_join, parse_url, update_query,
    http_request, dns_lookup, reverse_dns, ping, port_open, run_concurrent,
    DEFAULT_HEADERS,
)
from pentest import recon_tools, vuln_scanner, vuln_exploits, waf_bypass


def _ok(data=None, msg="ok"):
    return {"ok": True, "msg": msg, "data": data}

def _err(msg, code=400):
    from fastapi.responses import JSONResponse
    return JSONResponse({"ok": False, "msg": msg}, status_code=code)


# =====================================================================
# 技能指令系统 (在对话中通过 >> 前缀调用渗透工具)
# =====================================================================
SKILL_HELP = """
╔══════════════════════════════════════════════════════════════╗
║              🛡️ 白帽渗透测试技能指令速查表                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                                ║
║  【侦察类】                                                    ║
║  >>帮助              显示本帮助                                 ║
║  >>whois <域名>      Whois/DNS/Ping 信息收集                   ║
║  >>子域名 <域名>     子域名爆破 + CT证书透明度日志              ║
║  >>端口 <主机>       TCP端口扫描 + Banner抓取                  ║
║  >>目录 <URL>        目录/敏感文件/路径爆破                     ║
║  >>指纹 <URL>        CMS/中间件/WAF 指纹识别                    ║
║  >>爆破 <URL>        HTTP弱口令爆破 (表单/Basic)               ║
║  >>全量侦察 <目标>   一键全量 (子域+端口+目录+指纹)            ║
║                                                                ║
║  【漏洞扫描类】                                                ║
║  >>扫描 <URL>                    一键全部漏洞扫描              ║
║  >>扫描 sqli <URL>               SQL注入检测                   ║
║  >>扫描 xss <URL>                XSS检测                       ║
║  >>扫描 rce <URL>                命令注入/RCE检测              ║
║  >>扫描 ssti <URL>               SSTI模板注入检测              ║
║  >>扫描 ssrf <URL> <参数名>      SSRF检测                      ║
║  >>扫描 lfi <URL> <参数名>       目录遍历/LFI检测              ║
║  >>扫描 xxe <URL>                XXE检测                       ║
║  >>扫描 unauth <URL>             未授权端点检测                ║
║  >>扫描 middleware <URL>         中间件解析漏洞                ║
║  >>扫描 info <URL>               信息泄漏检测                  ║
║  >>扫描 deserialize <URL>        反序列化CVE指纹               ║
║  >>扫描 upload <上传URL>         文件上传漏洞检测              ║
║                                                                ║
║  【漏洞利用库】                                                ║
║  >>利用                显示全部漏洞利用链                      ║
║  >>利用 SQLi           SQL注入完整利用链 (8步+Payload)         ║
║  >>利用 XSS            XSS 8阶段利用链                         ║
║  >>利用 SSRF           SSRF 8阶段利用链                        ║
║  >>利用 RCE            RCE 7步利用链                           ║
║  >>利用 SSTI           SSTI 7种模板引擎RCE                    ║
║  >>利用 FileUpload     19种上传绕过技术                        ║
║  >>利用 CVE            Shiro550/Log4Shell等一键CVE             ║
║                                                                ║
║  【WAF绕过库】                                                 ║
║  >>绕过                显示全部WAF绕过技术                     ║
║  >>绕过 SQLi           SQL注入WAF绕过                          ║
║  >>绕过 XSS            XSS WAF绕过                             ║
║  >>绕过 RCE            RCE WAF绕过                             ║
║  >>绕过 SSTI           SSTI WAF绕过                            ║
║  >>绕过 SSRF           SSRF WAF绕过                            ║
║  >>绕过 Upload         上传WAF绕过                             ║
║  >>绕过 LFI            LFI WAF绕过                             ║
║  >>绕过 通用           通用HTTP层绕过                          ║
║  >>绕过生成 <payload>  自动生成WAF绕过变体                     ║
║                                                                ║
║  >>工具                显示工具API速查表                       ║
╚══════════════════════════════════════════════════════════════╝
"""


def _execute_skill(cmd_line: str):
    """解析并执行 >> 前缀的技能指令, 返回文本结果或 None"""
    text = cmd_line[2:].strip()
    if not text:
        return SKILL_HELP

    parts = text.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    try:
        # ---- 帮助 ----
        if cmd in ("帮助", "help", "?", "h"):
            return SKILL_HELP

        # ---- 工具列表 ----
        if cmd in ("工具", "tools", "tool"):
            return _skill_tools_list()

        # ---- Whois ----
        if cmd in ("whois", "域名信息"):
            if not arg:
                return "❌ 用法: >>whois <域名>\n例: >>whois baidu.com"
            return _skill_whois(arg)

        # ---- 子域名 ----
        if cmd in ("子域名", "subdomain", "sub"):
            if not arg:
                return "❌ 用法: >>子域名 <域名>\n例: >>子域名 baidu.com"
            return _skill_subdomain(arg)

        # ---- 端口扫描 ----
        if cmd in ("端口", "port", "端口扫描"):
            if not arg:
                return "❌ 用法: >>端口 <主机/IP>\n例: >>端口 127.0.0.1"
            return _skill_port(arg)

        # ---- 目录扫描 ----
        if cmd in ("目录", "dir", "目录扫描"):
            if not arg:
                return "❌ 用法: >>目录 <URL>\n例: >>目录 http://target.com"
            return _skill_dir(arg)

        # ---- 指纹 ----
        if cmd in ("指纹", "fingerprint", "fp"):
            if not arg:
                return "❌ 用法: >>指纹 <URL>\n例: >>指纹 http://target.com"
            return _skill_fingerprint(arg)

        # ---- 爆破 ----
        if cmd in ("爆破", "brute"):
            if not arg:
                return "❌ 用法: >>爆破 <URL>\n例: >>爆破 http://target.com/login.php"
            return _skill_brute(arg)

        # ---- 全量侦察 ----
        if cmd in ("全量侦察", "full", "全量"):
            if not arg:
                return "❌ 用法: >>全量侦察 <目标>\n例: >>全量侦察 baidu.com"
            return _skill_full_recon(arg)

        # ---- 漏洞扫描 ----
        if cmd in ("扫描", "scan"):
            return _skill_scan(arg)

        # ---- 漏洞利用库 ----
        if cmd in ("利用", "exploit", "exp"):
            return _skill_exploit(arg)

        # ---- WAF绕过 ----
        if cmd in ("绕过", "bypass"):
            return _skill_bypass(arg)

        # ---- 绕过变体生成 ----
        if cmd in ("绕过生成", "bypass_gen", "变体"):
            if not arg:
                return "❌ 用法: >>绕过生成 <payload>\n例: >>绕过生成 1' union select 1,2,3--"
            return _skill_bypass_gen(arg)

        return f"❌ 未知指令: >>{cmd}\n输入 >>帮助 查看所有可用指令"

    except Exception as e:
        return f"❌ 执行失败: {e}"


def _skill_whois(target: str) -> str:
    r = recon_tools.whois_domain(target)
    lines = [f"🔍 Whois 信息收集: {target}", "=" * 55]
    lines.append(f"DNS解析: {', '.join(r.get('dns', [])) or '无'}")
    rev = r.get('reverse', {})
    if rev:
        lines.append(f"反向解析: {rev}")
    ping_ms = r.get('ping_ms')
    lines.append(f"TCP Ping: {f'{ping_ms:.1f} ms' if ping_ms else '超时'}")
    if r.get('rdap'):
        lines.append(f"RDAP: {r['rdap']}")
    return "\n".join(lines)


def _skill_subdomain(domain: str) -> str:
    lines = [f"🔍 子域名扫描: {domain}", "⏳ 正在扫描 (可能需要30s-2min)...", "=" * 55]
    results = recon_tools.subdomain_scan(domain)
    if not results:
        return f"🔍 子域名扫描: {domain}\n未发现子域名。"
    lines.append(f"发现 {len(results)} 个子域名:\n")
    lines.append(f"{'子域名':<35} {'IP':<16} {'状态':<6} {'标题'}")
    lines.append("-" * 90)
    for r in results[:200]:
        lines.append(f"{r.get('subdomain',''):<35} {str(r.get('ip','-')):<16} {str(r.get('status',0)):<6} {(r.get('title','') or '')[:40]}")
    if len(results) > 200:
        lines.append(f"\n... 共 {len(results)} 条,仅显示前200条")
    return "\n".join(lines)


def _skill_port(host: str) -> str:
    lines = [f"🔍 端口扫描: {host}", "⏳ 正在扫描 Top 80 端口...", "=" * 55]
    results = recon_tools.port_scan(host, top_n=80, max_workers=80, timeout=1.5, banner=True)
    if not results:
        return f"🔍 端口扫描: {host}\n无开放端口。"
    lines.append(f"开放端口 {len(results)} 个:\n")
    lines.append(f"{'端口':<8} {'服务':<15} {'Banner/指纹'}")
    lines.append("-" * 80)
    for r in results:
        banner = (r.get('banner', '') or '')[:60]
        lines.append(f"{r.get('port',''):<8} {r.get('service','-'):<15} {banner}")
    return "\n".join(lines)


def _skill_dir(url: str) -> str:
    lines = [f"🔍 目录扫描: {url}", "⏳ 正在爆破目录...", "=" * 55]
    results = recon_tools.dir_scan(url, max_workers=30, timeout=5)
    if not results:
        return f"🔍 目录扫描: {url}\n未发现有效路径。"
    lines.append(f"命中路径 {len(results)} 个:\n")
    lines.append(f"{'状态':<6} {'路径':<40} {'长度':<8} {'标题'}")
    lines.append("-" * 85)
    for r in results[:200]:
        lines.append(f"{r.get('status',''):<6} {r.get('path',''):<40} {str(r.get('length',0)):<8} {(r.get('title','') or '')[:30]}")
    if len(results) > 200:
        lines.append(f"\n... 共 {len(results)} 条,仅显示前200条")
    return "\n".join(lines)


def _skill_fingerprint(url: str) -> str:
    r = recon_tools.fingerprint(url)
    lines = [f"🔍 指纹识别: {url}", "=" * 55]
    lines.append(f"HTTP状态: {r.get('status', 0)}")
    lines.append(f"标题: {r.get('title', '-')}")
    lines.append(f"Server: {r.get('server_header', '-')}")
    lines.append(f"X-Powered-By: {r.get('x_powered', '-')}")
    cms = r.get('cms', [])
    lines.append(f"CMS: {', '.join(cms) if cms else '未识别'}")
    mw = r.get('middleware', [])
    lines.append(f"中间件: {', '.join(mw) if mw else '未识别'}")
    waf = r.get('waf', [])
    lines.append(f"WAF: {', '.join(waf) if waf else '未检测到'}")
    cookies = r.get('cookies', [])
    if cookies:
        lines.append(f"Cookies: {', '.join(cookies)}")
    probes = r.get('extra_probes', [])
    if probes:
        lines.append(f"\n额外探测:")
        for p in probes:
            lines.append(f"  ✅ {p}")
    return "\n".join(lines)


def _skill_brute(url: str) -> str:
    lines = [f"🔍 弱口令爆破: {url}", "⏳ 正在爆破 (默认字典)...", "=" * 55]
    result = recon_tools.brute_http_form(url, stop_on_first=True, max_workers=10)
    hits = result if isinstance(result, list) else result.get('results', [])
    if not hits:
        return f"🔍 弱口令爆破: {url}\n未爆破出有效凭据。\n提示: 默认字典较小,可尝试自定义账号密码。"
    lines.append(f"爆破命中 {len(hits)} 条:\n")
    lines.append(f"{'账号':<20} {'密码':<20} {'状态':<6} {'长度'}")
    lines.append("-" * 60)
    for h in hits:
        lines.append(f"{h.get('username',h.get('user','')):<20} {h.get('password',h.get('pass','')):<20} {h.get('status',''):<6} {h.get('length','')}")
    return "\n".join(lines)


def _skill_full_recon(target: str) -> str:
    lines = [f"🔍 全量侦察: {target}", "⏳ 正在执行 (预计2-5分钟,请勿刷新)...", "=" * 55]
    opts = {"sub": True, "port": True, "dir": True, "port_top_n": 50}
    data = recon_tools.full_recon(target, opts=opts)
    lines.append("\n--- Whois / Ping ---")
    wh = data.get('whois', {})
    if wh:
        lines.append(f"DNS: {', '.join(wh.get('dns', []))}")
        lines.append(f"Ping: {wh.get('ping_ms', '超时')}")
    subs = data.get('subdomains', [])
    lines.append(f"\n--- 子域名 ({len(subs)} 个) ---")
    for s in subs[:30]:
        lines.append(f"  {s.get('subdomain',''):<30} {s.get('ip',''):<16} {s.get('status',0)}")
    if len(subs) > 30:
        lines.append(f"  ... 共{len(subs)}条")
    ports = data.get('ports', [])
    lines.append(f"\n--- 开放端口 ({len(ports)} 个) ---")
    for p in ports[:30]:
        lines.append(f"  {p.get('port',''):<8} {p.get('service','-'):<15} {(p.get('banner','') or '')[:40]}")
    if len(ports) > 30:
        lines.append(f"  ... 共{len(ports)}条")
    fp = data.get('fingerprint', {})
    lines.append(f"\n--- 指纹 ---")
    if fp:
        lines.append(f"  Server: {fp.get('server_header','-')}")
        lines.append(f"  CMS: {', '.join(fp.get('cms',[])) or '-'}")
        lines.append(f"  WAF: {', '.join(fp.get('waf',[])) or '-'}")
    dirs = data.get('dirscan', [])
    lines.append(f"\n--- 目录/敏感文件 ({len(dirs)} 个) ---")
    for d in dirs[:30]:
        lines.append(f"  {d.get('status',''):<6} {d.get('path','')}")
    if len(dirs) > 30:
        lines.append(f"  ... 共{len(dirs)}条")
    return "\n".join(lines)


def _skill_scan(arg: str) -> str:
    parts = arg.split(None, 1)
    # 判断第一个参数是否是扫描类型
    scan_types = {"sqli", "xss", "rce", "ssti", "ssrf", "lfi", "xxe", "unauth",
                  "middleware", "info", "deserialize", "upload", "all", "全部"}
    if parts and parts[0].lower() in scan_types:
        tp = parts[0].lower()
        rest = parts[1].strip() if len(parts) > 1 else ""
    else:
        tp = "all"
        rest = arg.strip()

    if not rest:
        return ("❌ 用法:\n"
                "  >>扫描 <URL>           一键全部扫描\n"
                "  >>扫描 sqli <URL>      SQL注入\n"
                "  >>扫描 xss <URL>       XSS\n"
                "  >>扫描 rce <URL>       RCE\n"
                "  >>扫描 ssti <URL>      SSTI\n"
                "  >>扫描 ssrf <URL> <参数名>\n"
                "  >>扫描 lfi <URL> <参数名>\n"
                "  >>扫描 unauth <URL>    未授权\n"
                "  >>扫描 upload <上传URL>\n"
                "  类型: sqli/xss/rce/ssti/ssrf/lfi/xxe/unauth/middleware/info/deserialize/upload/all")

    # 解析参数
    url_parts = rest.split(None, 1)
    url = url_parts[0]
    param = url_parts[1].strip() if len(url_parts) > 1 else ""

    if tp == "all" or tp == "全部":
        lines = [f"💉 漏洞全量扫描: {url}", "⏳ 正在扫描所有漏洞类型 (可能需要1-3分钟)...", "=" * 55]
        extra = {}
        if param:
            extra["ssrf_param"] = param
            extra["lfi_param"] = param
        results = vuln_scanner.scan_all(url, params=extra, threads=10)
        return _format_scan_results(url, results, "全量扫描")

    name_map = {
        "sqli": "SQL注入", "xss": "XSS", "rce": "命令注入RCE", "ssti": "SSTI模板注入",
        "ssrf": "SSRF", "lfi": "LFI/目录遍历", "xxe": "XXE",
        "unauth": "未授权端点", "middleware": "中间件解析漏洞",
        "info": "信息泄漏", "deserialize": "反序列化/CVE指纹",
    }

    fn_map = {
        "sqli": lambda: vuln_scanner.scan_sqli(url),
        "xss": lambda: vuln_scanner.scan_xss(url),
        "rce": lambda: vuln_scanner.scan_rce(url),
        "ssti": lambda: vuln_scanner.scan_ssti(url),
        "ssrf": lambda: vuln_scanner.scan_ssrf(url, param or "url"),
        "lfi": lambda: vuln_scanner.scan_lfi(url, param or "file"),
        "xxe": lambda: vuln_scanner.scan_xxe(url),
        "unauth": lambda: vuln_scanner.scan_unauth_endpoints(url),
        "middleware": lambda: vuln_scanner.scan_middleware(url),
        "info": lambda: vuln_scanner.scan_info(url),
        "deserialize": lambda: vuln_scanner.scan_deserialize(url),
    }

    if tp == "upload":
        lines = [f"💉 文件上传漏洞检测: {url}", "⏳ 正在检测...", "=" * 55]
        results = vuln_scanner.scan_file_upload(url, "file")
        return _format_scan_results(url, results, "文件上传")

    if tp in fn_map:
        lines = [f"💉 {name_map.get(tp, tp)} 检测: {url}", "⏳ 正在检测...", "=" * 55]
        try:
            results = fn_map[tp]()
            return _format_scan_results(url, results, name_map.get(tp, tp))
        except Exception as e:
            return f"💉 {name_map.get(tp, tp)} 检测: {url}\n❌ 执行失败: {e}"

    return f"❌ 不支持的扫描类型: {tp}"


def _format_scan_results(url: str, results, scan_name: str) -> str:
    if not results:
        return f"💉 {scan_name}: {url}\n✅ 扫描完成,未检测到漏洞。"
    vuln = [r for r in results if isinstance(r, dict) and r.get('is_vuln')]
    info = [r for r in results if isinstance(r, dict) and not r.get('is_vuln')]
    lines = [f"💉 {scan_name}: {url}", "=" * 55]
    lines.append(f"漏洞命中: {len(vuln)} 个 | 信息/失败: {len(info)} 个\n")
    if vuln:
        lines.append("🔴 漏洞命中:")
        lines.append(f"{'风险':<8} {'名称':<25} {'证据'}")
        lines.append("-" * 80)
        for v in vuln:
            sev = v.get('severity', 'info').upper()
            name = (v.get('name') or v.get('scanner') or v.get('category') or '')[:24]
            evidence = (v.get('evidence') or v.get('error') or '')[:50]
            lines.append(f"{sev:<8} {name:<25} {evidence}")
    if info:
        lines.append(f"\n⚪ 信息/未命中 ({len(info)} 条,显示前20):")
        for v in info[:20]:
            name = (v.get('name') or v.get('scanner') or v.get('category') or '')[:24]
            lines.append(f"  - {name}")
    return "\n".join(lines)


def _skill_exploit(arg: str) -> str:
    q = arg.strip() if arg else None
    data = vuln_exploits.get_exploit(q)
    if not data:
        return f"📚 漏洞利用库: 未匹配到 '{q}'\n可搜索: SQLi / XSS / SSRF / RCE / SSTI / FileUpload / CVE"
    lines = [f"📚 漏洞利用代码库 ({len(data)} 条匹配)", "=" * 55]
    for i, e in enumerate(data, 1):
        lines.append(f"\n{'─'*55}")
        lines.append(f"[{i}] {e.get('title', '')}")
        sev = e.get('severity', '')
        cvss = e.get('cvss', '')
        cve = e.get('cve', '')
        lines.append(f"  风险: {sev} | CVSS: {cvss} | {cve} | 分类: {e.get('category','')}")
        if e.get('affected'):
            lines.append(f"  影响范围: {e['affected']}")
        if e.get('overview'):
            lines.append(f"  概述: {e['overview']}")
        if e.get('detection'):
            lines.append(f"  检测方式: {e['detection']}")
        steps = e.get('exploit_steps', [])
        if steps:
            lines.append(f"  📋 利用步骤 ({len(steps)} 步):")
            for s in steps:
                lines.append(f"    • {s}")
        payloads = e.get('payloads', {})
        if payloads:
            lines.append(f"  💉 Payload 字典 ({len(payloads)} 个):")
            for k, v in payloads.items():
                lines.append(f"    [{k}]")
                lines.append(f"      {v}")
        techs = e.get('techniques', [])
        if techs:
            lines.append(f"  🔧 利用技术 ({len(techs)} 项):")
            for t in techs:
                lines.append(f"    ▸ {t.get('title','')}")
                if t.get('principle'):
                    lines.append(f"      原理: {t['principle']}")
                if t.get('example'):
                    lines.append(f"      示例: {t['example']}")
        bypass = e.get('bypass_tips', [])
        if bypass:
            lines.append(f"  🔥 WAF绕过提示 ({len(bypass)} 条):")
            for b in bypass:
                lines.append(f"    • {b}")
        post = e.get('post_exploitation', [])
        if post:
            lines.append(f"  📌 后利用 ({len(post)} 条):")
            for p in post:
                lines.append(f"    • {p}")
        refs = e.get('references', [])
        if refs:
            lines.append(f"  🔗 参考:")
            for r in refs:
                lines.append(f"    {r}")
    return "\n".join(lines)


def _skill_bypass(arg: str) -> str:
    q = arg.strip() if arg else None
    cats = waf_bypass.get_bypass(q)
    if not cats:
        return f"🔥 WAF绕过库: 未匹配到 '{q}'\n可搜索: 通用 / SQLi / XSS / RCE / SSTI / SSRF / Upload / LFI"
    lines = [f"🔥 WAF绕过技术库 ({len(cats)} 个分类)", "=" * 55]
    for c in cats:
        lines.append(f"\n{'─'*55}")
        lines.append(f"📂 {c.get('name', '')}")
        if c.get('desc'):
            lines.append(f"  {c['desc']}")
        techs = c.get('techniques', [])
        if techs:
            lines.append(f"  🔧 绕过技术 ({len(techs)} 项):")
            for t in techs:
                lines.append(f"    ▸ {t.get('title','')}")
                if t.get('principle'):
                    lines.append(f"      原理: {t['principle']}")
                if t.get('example'):
                    lines.append(f"      示例: {t['example']}")
        payloads = c.get('payloads', [])
        if payloads:
            show = payloads[:30]
            lines.append(f"  💉 Payload 样例 ({len(payloads)} 个,显示前{len(show)}):")
            for p in show:
                lines.append(f"    {p}")
    return "\n".join(lines)


def _skill_bypass_gen(payload: str) -> str:
    variants = waf_bypass.generate_payload_variants(payload, max_depth=2)
    lines = [f"🔥 WAF绕过变体生成", f"原始Payload: {payload}", f"生成变体: {len(variants)} 个", "=" * 55]
    for i, v in enumerate(variants[:200], 1):
        lines.append(f"  [{i:>3}] {v}")
    if len(variants) > 200:
        lines.append(f"\n... 共 {len(variants)} 个变体,仅显示前200个")
    lines.append("\n💡 提示: 逐个尝试,观察WAF拦截与响应变化定位可绕过的形式。")
    return "\n".join(lines)


def _skill_tools_list() -> str:
    lines = ["📦 渗透工具API速查表", "=" * 55]
    lines.append("\n所有API前缀: /api/pentest/")
    lines.append("返回格式: { ok, msg, disclaimer, data }")
    lines.append("\n🔍 侦察 Recon:")
    lines.append("  POST /api/pentest/recon/whois       {target}")
    lines.append("  POST /api/pentest/recon/subdomain   {target, sub_list?, workers?}")
    lines.append("  POST /api/pentest/recon/port        {target, ports?, top_n?, workers?}")
    lines.append("  POST /api/pentest/recon/dir         {target, paths?, ext?, workers?}")
    lines.append("  POST /api/pentest/recon/fingerprint {target}")
    lines.append("  POST /api/pentest/recon/brute       {url, mode, users?, passwords?}")
    lines.append("  POST /api/pentest/recon/full        {target, sub?, port?, dir?}")
    lines.append("\n💉 漏洞扫描 Scan:")
    lines.append("  POST /api/pentest/scan  {target, type, param?, upload_url?, threads?}")
    lines.append("  类型: sqli/xss/rce/ssti/ssrf/lfi/xxe/unauth/middleware/info/deserialize/upload/all")
    lines.append("\n⚔️ 漏洞利用库 Exploits:")
    lines.append("  GET  /api/pentest/exploits           全量")
    lines.append("  POST /api/pentest/exploits           {query: 'SQLi'}")
    lines.append("\n🔥 WAF绕过 Bypass:")
    lines.append("  GET  /api/pentest/bypass             全量")
    lines.append("  POST /api/pentest/bypass             {category?, payload?, depth?}")
    return "\n".join(lines)


# ============ 1. 侦察 Recon 工具 ============
@app.post("/api/pentest/recon/whois")
async def api_recon_whois(req: Request):
    data = await req.json()
    domain = (data.get("target") or data.get("domain") or "").strip()
    if not domain:
        return _err("请输入目标域名")
    return _ok(data=recon_tools.whois_domain(domain))


@app.post("/api/pentest/recon/subdomain")
async def api_recon_subdomain(req: Request):
    data = await req.json()
    domain = (data.get("target") or data.get("domain") or "").strip()
    subs_raw = data.get("sub_list") or None
    subs = [s.strip() for s in subs_raw.split(",")] if isinstance(subs_raw, str) else subs_raw
    workers = int(data.get("workers") or 50)
    if not domain:
        return _err("请输入目标根域名")
    return _ok(data=recon_tools.subdomain_scan(domain, sub_list=subs, max_workers=workers))


@app.post("/api/pentest/recon/port")
async def api_recon_port(req: Request):
    data = await req.json()
    host = (data.get("target") or data.get("host") or "").strip()
    ports = data.get("ports") or None
    if isinstance(ports, str) and ports:
        if "-" in ports:
            a, b = ports.split("-", 1)
            ports = list(range(int(a), int(b) + 1))
        else:
            ports = [int(x) for x in ports.replace("，", ",").split(",") if x.strip()]
    top_n = int(data.get("top_n") or 80)
    workers = int(data.get("workers") or 80)
    timeout = float(data.get("timeout") or 1.5)
    banner = bool(data.get("banner") if data.get("banner") is not None else True)
    if not host:
        return _err("请输入目标主机/IP")
    return _ok(data=recon_tools.port_scan(host, ports=ports, top_n=top_n,
                                          max_workers=workers, timeout=timeout, banner=banner))


@app.post("/api/pentest/recon/dir")
async def api_recon_dir(req: Request):
    data = await req.json()
    url = (data.get("target") or data.get("url") or "").strip()
    if not url:
        return _err("请输入目标URL")
    paths_raw = data.get("paths") or None
    paths = [p.strip() for p in paths_raw.replace("，", ",").split(",") if p.strip()] if isinstance(paths_raw, str) else None
    ext_raw = data.get("ext") or None
    ext = [e.strip() for e in ext_raw.replace("，", ",").split(",") if e.strip()] if isinstance(ext_raw, str) else None
    workers = int(data.get("workers") or 30)
    timeout = float(data.get("timeout") or 5)
    return _ok(data=recon_tools.dir_scan(url, paths=paths, ext=ext, max_workers=workers, timeout=timeout))


@app.post("/api/pentest/recon/fingerprint")
async def api_recon_fp(req: Request):
    data = await req.json()
    url = (data.get("target") or data.get("url") or "").strip()
    if not url:
        return _err("请输入目标URL")
    return _ok(data=recon_tools.fingerprint(url))


@app.post("/api/pentest/recon/brute")
async def api_recon_brute(req: Request):
    """HTTP 表单/Basic 弱口令爆破"""
    data = await req.json()
    mode = (data.get("mode") or "form").lower()
    url = (data.get("url") or "").strip()
    if not url:
        return _err("请输入目标URL")
    users_raw = data.get("users") or None
    if isinstance(users_raw, str):
        users = [u.strip() for u in users_raw.replace("，", ",").split(",") if u.strip()]
    else:
        users = None
    pws_raw = data.get("passwords") or None
    if isinstance(pws_raw, str):
        pws = [p.strip() for p in pws_raw.replace("，", ",").split(",") if p.strip()]
    else:
        pws = None
    workers = int(data.get("workers") or 10)
    stop_first = bool(data.get("stop_first") if data.get("stop_first") is not None else True)
    if mode == "basic":
        result = recon_tools.brute_http_basic(url, users=users, pws=pws, max_workers=workers)
    else:
        lp = data.get("login_params") or {}
        if isinstance(lp, str):
            lp = {"fail_keyword": lp}
        result = recon_tools.brute_http_form(
            url, usernames=users, passwords=pws,
            login_params=lp, method=data.get("method") or "POST",
            max_workers=workers, stop_on_first=stop_first,
        )
    return _ok(data={"mode": mode, "results": result})


@app.post("/api/pentest/recon/full")
async def api_recon_full(req: Request):
    """一键大礼包: 子域名 + 端口 + 目录 + 指纹"""
    data = await req.json()
    target = (data.get("target") or "").strip()
    if not target:
        return _err("请输入目标")
    opts = {
        "sub": bool(data.get("sub") if data.get("sub") is not None else True),
        "port": bool(data.get("port") if data.get("port") is not None else True),
        "dir": bool(data.get("dir") if data.get("dir") is not None else True),
        "port_top_n": int(data.get("port_top_n") or 50),
    }
    return _ok(data=recon_tools.full_recon(target, opts=opts))


# ============ 2. 漏洞扫描 Vuln Scanner ============
def _run_scan(name, fn, url, *args, **kwargs):
    try:
        data = fn(url, *args, **kwargs)
        return {"scanner": name, "ok": True, "results": data}
    except Exception as e:
        return {"scanner": name, "ok": False, "error": str(e), "results": []}


@app.post("/api/pentest/scan")
async def api_scan(req: Request):
    """单个扫描器调用: url + 类型 (sqli/xss/rce/ssti/lfi/ssrf/xxe/unauth/middleware/info/deserialize/upload/all)"""
    data = await req.json()
    url = (data.get("target") or data.get("url") or "").strip()
    if not url:
        return _err("请输入目标URL")
    tp = (data.get("type") or "all").lower()
    mapping = {
        "sqli": lambda: _run_scan("SQL注入", vuln_scanner.scan_sqli, url),
        "xss":  lambda: _run_scan("XSS", vuln_scanner.scan_xss, url),
        "rce":  lambda: _run_scan("命令注入RCE", vuln_scanner.scan_rce, url),
        "ssti": lambda: _run_scan("SSTI", vuln_scanner.scan_ssti, url),
        "lfi":  lambda: _run_scan("LFI/目录遍历", vuln_scanner.scan_lfi, url, data.get("param") or "file"),
        "ssrf": lambda: _run_scan("SSRF", vuln_scanner.scan_ssrf, url, data.get("param") or "url"),
        "xxe":  lambda: _run_scan("XXE", vuln_scanner.scan_xxe, url),
        "unauth": lambda: _run_scan("未授权端点", vuln_scanner.scan_unauth_endpoints, url),
        "middleware": lambda: _run_scan("中间件解析漏洞", vuln_scanner.scan_middleware, url),
        "info": lambda: _run_scan("信息泄漏", vuln_scanner.scan_info, url),
        "deserialize": lambda: _run_scan("反序列化/CVE指纹", vuln_scanner.scan_deserialize, url),
    }
    if tp == "upload":
        return _ok(data=_run_scan("文件上传", vuln_scanner.scan_file_upload,
                                   data.get("upload_url") or url,
                                   data.get("file_field") or "file",
                                   data.get("submit_field"),
                                   data.get("success_keyword") or "",
                                   data.get("access_prefix") or ""))
    if tp == "all":
        extra = {}
        if data.get("ssrf_param"):
            extra["ssrf_param"] = data.get("ssrf_param")
        if data.get("lfi_param"):
            extra["lfi_param"] = data.get("lfi_param")
        if data.get("xxe_url"):
            extra["xxe_url"] = data.get("xxe_url")
        if data.get("upload_url"):
            extra["upload_url"] = data.get("upload_url")
            extra["file_field"] = data.get("file_field") or "file"
            extra["access_prefix"] = data.get("access_prefix") or ""
        return _ok(data={"all_scan": True,
                         "results": vuln_scanner.scan_all(url, params=extra,
                                                           threads=int(data.get("threads") or 10))})
    if tp in mapping:
        return _ok(data=mapping[tp]())
    return _err(f"不支持的扫描类型: {tp}。支持: " + ",".join(list(mapping.keys()) + ["upload", "all"]))


# ============ 3. 漏洞利用代码库 Exploits ============
@app.get("/api/pentest/exploits")
@app.post("/api/pentest/exploits")
async def api_exploits(req: Request = None):
    """获取漏洞利用代码库 + 利用链细节"""
    q = None
    if req is not None:
        try:
            if req.method == "POST":
                data = await req.json()
                q = data.get("query") or data.get("category")
            else:
                q = dict(req.query_params).get("q") or dict(req.query_params).get("category")
        except Exception:
            pass
    data = vuln_exploits.get_exploit(q)
    return _ok(data={
        "count": len(data),
        "query": q,
        "list": data,
    })


# ============ 4. WAF绕过 Payload + 技术库 ============
@app.get("/api/pentest/bypass")
@app.post("/api/pentest/bypass")
async def api_bypass(req: Request = None):
    q = None
    base_payload = None
    max_depth = 2
    if req is not None:
        try:
            if req.method == "POST":
                data = await req.json()
                q = data.get("category")
                base_payload = data.get("payload")
                max_depth = int(data.get("depth") or 2)
            else:
                qp = dict(req.query_params)
                q = qp.get("category")
                base_payload = qp.get("payload")
                max_depth = int(qp.get("depth") or 2)
        except Exception:
            pass
    result = {"categories": waf_bypass.get_bypass(q)}
    if base_payload:
        variants = waf_bypass.generate_payload_variants(base_payload, max_depth=max_depth)
        result["generated_variants_count"] = len(variants)
        result["generated_variants"] = variants[:500]  # 返回前500个避免过大
    return _ok(data=result)


# ============ 5. 工具速查列表 ============
@app.get("/api/pentest/tools")
async def api_tools_list():
    return _ok(data={
        "recon": [
            {"id": "whois", "name": "Whois/信息收集", "params": ["target"]},
            {"id": "subdomain", "name": "子域名爆破 + CT日志", "params": ["target", "sub_list?", "workers?"]},
            {"id": "port", "name": "端口扫描 + Banner/服务", "params": ["target", "ports?", "top_n?", "workers?"]},
            {"id": "dir", "name": "目录/敏感文件爆破", "params": ["target", "paths?", "ext?", "workers?"]},
            {"id": "fingerprint", "name": "CMS/中间件/WAF指纹", "params": ["target"]},
            {"id": "brute", "name": "HTTP 弱口令爆破", "params": ["url", "mode:form|basic", "users?", "passwords?"]},
            {"id": "full", "name": "一键全量侦察大礼包", "params": ["target", "sub?", "port?", "dir?"]},
        ],
        "scan": [
            {"id": "sqli", "name": "SQL注入检测 (报错/布尔盲注/时间盲注)"},
            {"id": "xss",  "name": "反射型XSS检测"},
            {"id": "rce",  "name": "命令注入/RCE检测"},
            {"id": "ssti", "name": "SSTI模板注入检测"},
            {"id": "ssrf", "name": "SSRF服务端请求伪造 (加?param=参数名)"},
            {"id": "xxe",  "name": "XXE XML外部实体注入"},
            {"id": "lfi",  "name": "目录遍历/LFI (加?param=参数名)"},
            {"id": "unauth", "name": "Spring Actuator/Swagger/Nacos 未授权端点"},
            {"id": "middleware", "name": "Nginx/Apache/IIS 中间件解析漏洞"},
            {"id": "info", "name": "信息泄漏 (IP/邮箱/JWT/密钥/注释)"},
            {"id": "deserialize", "name": "Shiro/Fastjson/Thinkphp CVE指纹快速检测"},
            {"id": "upload", "name": "文件上传绕过检测"},
            {"id": "all",  "name": "一键全部扫描"},
        ],
        "exploits": [
            {"id": "SQLi", "desc": "SQL注入完整利用链 8步骤 + 14种DB Payload"},
            {"id": "XSS",  "desc": "XSS 8阶段利用 (CSP/Bypass/Cookie/CSRF/Electron RCE)"},
            {"id": "SSRF", "desc": "SSRF 8阶段利用 (云元数据/Gopher/Dict 内网服务攻击链)"},
            {"id": "RCE",  "desc": "RCE 7步骤利用链 (反弹shell/外带/提权信息集合)"},
            {"id": "SSTI", "desc": "Jinja2/Thymeleaf/Freemarker/Twig/Smarty/Velocity 全栈RCE链"},
            {"id": "FileUpload", "desc": "19种上传绕过技术 + 解析漏洞 + 图片马"},
            {"id": "CVE",  "desc": "Shiro550/Fastjson/Thinkphp/Spring4Shell/Log4Shell/Nacos 一键CVE"},
        ],
        "bypass": [
            {"id": "通用HTTP层", "desc": "编码/HPP/分块/请求走私/缓存投毒/XFF伪装"},
            {"id": "SQLi", "desc": "空格/关键字/引号/逗号/sleep/函数 12+绕过 + Payload"},
            {"id": "XSS",  "desc": "CSP/BaseURI/引号/关键字/UTF7/HTML容错 10+ + Payload"},
            {"id": "RCE",  "desc": "分隔符/空格/关键字/base64/disable_functions 9+ + Payload"},
            {"id": "SSTI", "desc": "点/下划线/引号/括号/关键字/{{ 7绕过 + Payload"},
            {"id": "SSRF", "desc": "IP变换/协议/302/DNS Rebinding/短网址 7绕过 + Payload"},
            {"id": "Upload", "desc": "19种上传绕过技术全集合"},
            {"id": "LFI",    "desc": "多重编码/UTF-8 overlong/一次删除/流包装器 PHP 9绕过"},
            {"id": "generator", "desc": "WAF Payload 自动生成变体工具 generate_payload_variants()"},
        ],
    })


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
