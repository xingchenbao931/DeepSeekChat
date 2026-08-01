# -*- coding: utf-8 -*-
"""
GGUF 模型下载脚本
从 ModelScope(魔搭社区)下载 Qwen2.5-3B-Instruct GGUF 量化模型
(兼容 DeepSeekChat 框架, 基于同一家族的 Qwen 架构)
"""
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

# ============ 模型信息 ============
# Qwen2.5-3B-Instruct - 阿里千问官方 3B 参数对话模型
# Q4_K_M 量化, 体积约 1.93GB, 适合 CPU 推理, 中文能力强
MODEL_NAME = "qwen2.5-3b-it-Q4_K_M-LOT.gguf"
MODEL_DISPLAY_NAME = "Qwen2.5-3B-Instruct Q4_K_M"
MODEL_SIZE_GB = "1.93 GB"

# ModelScope 下载地址 (国内速度快, 实测可用)
# 注意: 文件名中的下划线 _ 需要 URL 编码为 %5F
MODELSCOPE_URL = (
    "https://modelscope.cn/models/"
    "okwinds/Qwen2.5-3B-Instruct-GGUF-V3-LOT/"
    "resolve/master/"
    "qwen2.5-3b-it-Q4%5FK%5FM-LOT.gguf"
)

# 备用: 官方 Qwen GGUF 仓库 (如果上面不可用)
MODELSCOPE_URL_BACKUP = (
    "https://modelscope.cn/models/"
    "Qwen/Qwen2.5-3B-Instruct-GGUF/"
    "resolve/master/"
    "qwen2.5-3b-instruct-q4_k_m.gguf"
)
MODEL_NAME_BACKUP = "qwen2.5-3b-instruct-q4_k_m.gguf"

# ============ 下载目录 ============
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
TARGET_PATH = MODELS_DIR / MODEL_NAME


def download_with_progress(url: str, target: Path):
    """带进度条的文件下载"""
    print(f"\n正在从: {url}")
    print(f"下载到: {target}\n")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DeepSeekChat/1.0"})
        resp = urllib.request.urlopen(req, timeout=30)
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB

        with open(target, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = downloaded * 100 / total_size
                    size_mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    # 进度条
                    bar_len = 40
                    filled = int(bar_len * downloaded / total_size)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"\r  {bar} {percent:.1f}% ({size_mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)
                else:
                    size_mb = downloaded / (1024 * 1024)
                    print(f"\r  已下载: {size_mb:.1f} MB", end="", flush=True)

        print("\n\n下载完成!")
        return True

    except urllib.error.HTTPError as e:
        print(f"\n下载失败 (HTTP {e.code}): {e.reason}")
        return False
    except Exception as e:
        print(f"\n下载失败: {e}")
        if target.exists():
            target.unlink()
        return False


def main():
    print("=" * 60)
    print("  DeepSeekChat - 模型下载工具")
    print("=" * 60)
    print(f"  模型名称: {MODEL_DISPLAY_NAME}")
    print(f"  量化版本: Q4_K_M (约 {MODEL_SIZE_GB})")
    print(f"  适合 CPU 推理, 中文能力强")
    print("=" * 60)

    # 检查是否已存在 (支持多个可能的文件名)
    existing = None
    for f in MODELS_DIR.glob("*.gguf"):
        existing = f
        break
    if TARGET_PATH.exists():
        existing = TARGET_PATH

    if existing:
        size_mb = existing.stat().st_size / (1024 * 1024)
        print(f"\n模型已存在: {existing} ({size_mb:.1f} MB)")
        try:
            choice = input("是否重新下载? (y/N): ").strip().lower()
        except EOFError:
            choice = "n"
        if choice != "y":
            print("跳过下载。")
            return

    # 创建目录
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 尝试从 ModelScope 下载
    print("\n尝试从 ModelScope(魔搭社区)下载...")
    success = download_with_progress(MODELSCOPE_URL, TARGET_PATH)

    # 失败尝试备用地址
    if not success:
        print("\n主下载源失败, 尝试备用官方仓库...")
        backup_target = MODELS_DIR / MODEL_NAME_BACKUP
        success = download_with_progress(MODELSCOPE_URL_BACKUP, backup_target)
        if success:
            # 如果备用下载成功,重命名为标准名
            try:
                backup_target.rename(TARGET_PATH)
            except:
                pass

    if success:
        size_mb = TARGET_PATH.stat().st_size / (1024 * 1024)
        print(f"\n模型下载成功: {TARGET_PATH}")
        print(f"文件大小: {size_mb:.1f} MB")
        print("\n现在可以运行 启动.bat 启动 DeepSeekChat 了!")
    else:
        print("\n所有下载源均失败,请手动下载模型:")
        print(f"  1. 访问: https://modelscope.cn/models/okwinds/Qwen2.5-3B-Instruct-GGUF-V3-LOT/files")
        print(f"  2. 下载文件: {MODEL_NAME}")
        print(f"  3. 放置到: {MODELS_DIR}")
        sys.exit(1)


if __name__ == "__main__":
    main()
