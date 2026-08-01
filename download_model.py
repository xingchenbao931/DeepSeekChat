# -*- coding: utf-8 -*-
"""
DeepSeek GGUF 模型下载脚本
从 ModelScope(魔搭社区)下载 DeepSeek-R1-Distill-Qwen-1.5B GGUF 量化模型
"""
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

# ============ 模型信息 ============
# DeepSeek-R1-Distill-Qwen-1.5B 是 DeepSeek 推理模型的蒸馏版
# Q4_K_M 量化,体积约 1.1GB,适合 CPU 推理
MODEL_NAME = "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"

# ModelScope 下载地址 (国内速度快)
MODELSCOPE_URL = (
    "https://modelscope.cn/api/v1/models/"
    "AI-ModelScope/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/"
    "repo?Revision=master&FilePath="
    + MODEL_NAME
)

# HuggingFace 镜像备用地址
HF_MIRROR_URL = (
    "https://hf-mirror.com/Qwen/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/"
    "resolve/main/" + MODEL_NAME
)

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
    print(f"  模型名称: {MODEL_NAME}")
    print(f"  量化版本: Q4_K_M (约 1.1 GB)")
    print(f"  适合 CPU 推理,显存要求低")
    print("=" * 60)

    # 检查是否已存在
    if TARGET_PATH.exists():
        size_mb = TARGET_PATH.stat().st_size / (1024 * 1024)
        print(f"\n模型已存在: {TARGET_PATH} ({size_mb:.1f} MB)")
        choice = input("是否重新下载? (y/N): ").strip().lower()
        if choice != "y":
            print("跳过下载。")
            return

    # 创建目录
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 尝试从 ModelScope 下载
    print("\n尝试从 ModelScope(魔搭社区)下载...")
    success = download_with_progress(MODELSCOPE_URL, TARGET_PATH)

    # ModelScope 失败则尝试镜像
    if not success:
        print("\nModelScope 下载失败,尝试 HuggingFace 镜像...")
        success = download_with_progress(HF_MIRROR_URL, TARGET_PATH)

    if success:
        size_mb = TARGET_PATH.stat().st_size / (1024 * 1024)
        print(f"\n模型下载成功: {TARGET_PATH}")
        print(f"文件大小: {size_mb:.1f} MB")
        print("\n现在可以运行 启动.bat 启动 DeepSeekChat 了!")
    else:
        print("\n所有下载源均失败,请手动下载模型:")
        print(f"  1. 访问: https://modelscope.cn/models/AI-ModelScope/DeepSeek-R1-Distill-Qwen-1.5B-GGUF")
        print(f"  2. 下载文件: {MODEL_NAME}")
        print(f"  3. 放置到: {MODELS_DIR}")
        sys.exit(1)


if __name__ == "__main__":
    main()
