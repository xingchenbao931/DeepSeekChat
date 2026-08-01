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

    if not model_loaded or llm is None:
        return JSONResponse(
            {"error": "模型未加载,请先运行 download_model.py 下载模型"},
            status_code=503
        )

    data = await request.json()

    # ============ 双格式参数兼容 ============
    # 格式1: DeepSeekChat 风格 {message, history}
    # 格式2: QwenChat / OpenAI 风格 {messages[], temperature, max_tokens, system_prompt}
    if "messages" in data:
        raw_messages = data.get("messages", [])
        temperature = float(data.get("temperature", 0.7))
        max_tokens = int(data.get("max_tokens", 2048))
        system_prompt_override = data.get("system_prompt", None)
        use_delta = False  # OpenAI/QwenChat 风格返回 {content}
        messages = []
        if system_prompt_override:
            messages.append({"role": "system", "content": system_prompt_override})
        for m in raw_messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role in ("system", "user", "assistant"):
                messages.append({"role": role, "content": content})
        messages = _fit_messages_to_context(messages)
    else:
        user_input = data.get("message", "").strip()
        history = data.get("history", [])
        temperature = 0.7
        max_tokens = 2048
        use_delta = True  # DeepSeekChat 前端返回 {delta}? 不,统一返回 content, 前端兼容两种
        if not user_input:
            return JSONResponse({"error": "消息不能为空"}, status_code=400)
        messages = build_messages(user_input, history)

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
# 白帽渗透测试工具 API  (仅用于合法授权的安全测试)
# =====================================================================
from pentest import DISCLAIMER
from pentest.utils import (
    md5, b64e, b64d, url_encode, url_join, parse_url, update_query,
    http_request, dns_lookup, reverse_dns, ping, port_open, run_concurrent,
    DEFAULT_HEADERS,
)
from pentest import recon_tools, vuln_scanner, vuln_exploits, waf_bypass


def _ok(data=None, msg="ok"):
    return {"ok": True, "msg": msg, "disclaimer": DISCLAIMER.strip(), "data": data}

def _err(msg, code=400):
    from fastapi.responses import JSONResponse
    return JSONResponse({"ok": False, "msg": msg, "disclaimer": DISCLAIMER.strip()}, status_code=code)


@app.get("/api/pentest/disclaimer")
async def pentest_disclaimer():
    """获取法律免责声明"""
    return {"disclaimer": DISCLAIMER.strip()}


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
