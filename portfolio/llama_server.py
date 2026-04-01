"""Persistent llama-server manager for LLM inference.

Manages a SINGLE llama-server.exe process, swapping models as needed.
Only one model fits in VRAM (RTX 3080 10GB) at a time, so when a different
model is requested, the current server is stopped and a new one started.

Within a signal cycle, the same model gets queried for multiple tickers
without reloading. Between cycles, the model may swap.

Usage:
    from portfolio.llama_server import query_llama_server, stop_all_servers

    text = query_llama_server("ministral3", prompt)
    text = query_llama_server("qwen3", prompt)
"""

import logging
import os
import platform
import subprocess
import threading
import time

import requests as _requests

logger = logging.getLogger("portfolio.llama_server")

if platform.system() == "Windows":
    _LLAMA_SERVER = r"Q:\models\llama-cpp-bin\cuda13\llama-server.exe"
else:
    _LLAMA_SERVER = "/usr/local/bin/llama-server"

_MODEL_CONFIGS = {
    "ministral3": {
        "model": (
            r"Q:\models\ministral-3-8b-gguf\Ministral-3-8B-Instruct-2512-Q5_K_M.gguf"
            if platform.system() == "Windows"
            else "/home/deck/models/ministral-3-8b-gguf/Ministral-3-8B-Instruct-2512-Q5_K_M.gguf"
        ),
        "port": 8787,
    },
    "qwen3": {
        "model": (
            r"Q:\models\qwen3-8b-gguf\Qwen3-8B-Q4_K_M.gguf"
            if platform.system() == "Windows"
            else "/home/deck/models/qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf"
        ),
        "port": 8787,  # same port — only one runs at a time
    },
}

_server_proc = None       # current Popen
_active_model = None      # name of currently loaded model
_lock = threading.Lock()  # serializes all server operations


def _stop_current():
    """Stop the currently running server, if any."""
    global _server_proc, _active_model
    if _server_proc is not None:
        try:
            _server_proc.terminate()
            _server_proc.wait(timeout=10)
        except Exception:
            try:
                _server_proc.kill()
            except Exception:
                pass
        logger.info("llama-server stopped (was: %s)", _active_model)
        _server_proc = None
        _active_model = None


def _start_server(name):
    """Launch llama-server for the given model. Returns True if ready."""
    global _server_proc, _active_model
    cfg = _MODEL_CONFIGS.get(name)
    if cfg is None:
        return False
    if not os.path.exists(_LLAMA_SERVER) or not os.path.exists(cfg["model"]):
        logger.info("llama-server or model %s not found, skipping", name)
        return False

    # Stop any existing server first (different model, or dead process)
    _stop_current()
    time.sleep(1)  # brief pause for VRAM to be released

    try:
        proc = subprocess.Popen(
            [
                _LLAMA_SERVER,
                "-m", cfg["model"],
                "--port", str(cfg["port"]),
                "--host", "127.0.0.1",
                "-ngl", "99",
                "-t", "4",
                "-c", "4096",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 90
        while time.time() < deadline:
            if proc.poll() is not None:
                logger.warning("llama-server %s exited during startup (code %s)",
                               name, proc.returncode)
                return False
            try:
                r = _requests.get(f"http://127.0.0.1:{cfg['port']}/health", timeout=2)
                if r.status_code == 200:
                    logger.info("llama-server %s ready on port %d", name, cfg["port"])
                    _server_proc = proc
                    _active_model = name
                    return True
            except _requests.ConnectionError:
                time.sleep(1)
        logger.warning("llama-server %s startup timed out", name)
        proc.kill()
        return False
    except Exception as e:
        logger.warning("llama-server %s launch failed: %s", name, e)
        return False


def query_llama_server(name, prompt, n_predict=1024, temperature=0.0,
                       top_p=0.2, stop=None):
    """Query a persistent llama-server. Returns completion text or None.

    If a different model is currently loaded, swaps to the requested model.
    """
    cfg = _MODEL_CONFIGS.get(name)
    if cfg is None:
        return None

    with _lock:
        # Swap model if needed
        if _active_model != name or _server_proc is None or _server_proc.poll() is not None:
            if not _start_server(name):
                return None

    try:
        body = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop:
            body["stop"] = stop
        r = _requests.post(
            f"http://127.0.0.1:{cfg['port']}/completion",
            json=body,
            timeout=240,
        )
        if r.status_code == 200:
            return r.json().get("content", "").strip()
        logger.warning("llama-server %s returned %d", name, r.status_code)
        return None
    except Exception as e:
        logger.warning("llama-server %s query failed: %s", name, e)
        return None


def stop_server(name=None):
    """Stop the llama-server (optionally only if a specific model is loaded)."""
    with _lock:
        if name is None or _active_model == name:
            _stop_current()


def stop_all_servers():
    """Stop the llama-server regardless of which model is loaded."""
    with _lock:
        _stop_current()
