"""Unified persistent llama-server manager for ALL GGUF LLM inference.

Manages a SINGLE llama-server.exe process on port 8787, shared by both
main loop and metals loop. Swaps models as needed — only one 8B model
fits in VRAM (RTX 3080 10GB) at a time.

Cross-process coordination via file lock: both main.py and metals_loop.py
can call query_llama_server(), and the lock prevents simultaneous swaps.

Usage (from any process):
    from portfolio.llama_server import query_llama_server, stop_all_servers

    text = query_llama_server("ministral3", prompt)
    text = query_llama_server("qwen3", prompt)
    text = query_llama_server("ministral8_lora", prompt)
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

_PORT = 8787
_LOCK_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "llama_server.lock")
_PID_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "llama_server.pid")

_MODEL_CONFIGS = {
    "ministral3": {
        "model": (
            r"Q:\models\ministral-3-8b-gguf\Ministral-3-8B-Instruct-2512-Q5_K_M.gguf"
            if platform.system() == "Windows"
            else "/home/deck/models/ministral-3-8b-gguf/Ministral-3-8B-Instruct-2512-Q5_K_M.gguf"
        ),
        "extra_args": [],
    },
    "qwen3": {
        "model": (
            r"Q:\models\qwen3-8b-gguf\Qwen3-8B-Q4_K_M.gguf"
            if platform.system() == "Windows"
            else "/home/deck/models/qwen3-8b-gguf/Qwen3-8B-Q4_K_M.gguf"
        ),
        "extra_args": [],
    },
    "ministral8_lora": {
        "model": (
            r"Q:\models\ministral-8b-gguf\Ministral-8B-Instruct-2410-Q4_K_M.gguf"
            if platform.system() == "Windows"
            else "/home/deck/models/ministral-8b-gguf/Ministral-8B-Instruct-2410-Q4_K_M.gguf"
        ),
        "extra_args": [
            "--lora",
            r"Q:\models\cryptotrader-lm\cryptotrader-lm-lora.gguf"
            if platform.system() == "Windows"
            else "/home/deck/models/cryptotrader-lm/cryptotrader-lm-lora.gguf",
        ],
    },
}

# In-process state (each importing process tracks its own view)
_thread_lock = threading.Lock()
_local_proc = None       # Popen if this process started the server
_local_model = None      # model name this process loaded


def _kill_server_by_pid():
    """Kill any existing llama-server via saved PID file."""
    try:
        if os.path.exists(_PID_FILE):
            with open(_PID_FILE) as f:
                content = f.read().strip()
            if content:
                pid = int(content.split(":")[0])
                if platform.system() == "Windows":
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        capture_output=True, timeout=10,
                    )
                else:
                    os.kill(pid, 9)
                time.sleep(1)
    except Exception:
        pass


def _write_pid(proc, model_name):
    """Write server PID + model name so other processes know what's loaded."""
    try:
        os.makedirs(os.path.dirname(_PID_FILE), exist_ok=True)
        with open(_PID_FILE, "w") as f:
            f.write(f"{proc.pid}:{model_name}")
    except Exception:
        pass


def _read_pid_model():
    """Read PID + model from pid file. Returns (pid, model_name) or (None, None)."""
    try:
        if os.path.exists(_PID_FILE):
            with open(_PID_FILE) as f:
                content = f.read().strip()
            if ":" in content:
                pid_str, model = content.split(":", 1)
                return int(pid_str), model
    except Exception:
        pass
    return None, None


def _is_server_alive():
    """Check if llama-server is responding on the port."""
    try:
        r = _requests.get(f"http://127.0.0.1:{_PORT}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _stop_server():
    """Stop the currently running llama-server."""
    global _local_proc, _local_model

    # Kill by PID file first (covers cross-process case)
    _kill_server_by_pid()

    # Also kill our local ref if we started it
    if _local_proc is not None:
        try:
            _local_proc.terminate()
            _local_proc.wait(timeout=10)
        except Exception:
            try:
                _local_proc.kill()
            except Exception:
                pass
    _local_proc = None
    _local_model = None

    try:
        os.remove(_PID_FILE)
    except Exception:
        pass


def _start_server(name):
    """Launch llama-server with the given model. Returns True if ready."""
    global _local_proc, _local_model
    cfg = _MODEL_CONFIGS.get(name)
    if cfg is None:
        return False
    if not os.path.exists(_LLAMA_SERVER) or not os.path.exists(cfg["model"]):
        logger.info("llama-server or model %s not found", name)
        return False

    _stop_server()
    time.sleep(1)

    try:
        cmd = [
            _LLAMA_SERVER,
            "-m", cfg["model"],
            "--port", str(_PORT),
            "--host", "127.0.0.1",
            "-ngl", "99",
            "-t", "4",
            "-c", "4096",
        ] + cfg.get("extra_args", [])

        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        deadline = time.time() + 90
        while time.time() < deadline:
            if proc.poll() is not None:
                logger.warning("llama-server %s exited during startup (code %s)", name, proc.returncode)
                return False
            if _is_server_alive():
                logger.info("llama-server %s ready on port %d", name, _PORT)
                _local_proc = proc
                _local_model = name
                _write_pid(proc, name)
                return True
            time.sleep(1)
        logger.warning("llama-server %s startup timed out", name)
        proc.kill()
        return False
    except Exception as e:
        logger.warning("llama-server %s launch failed: %s", name, e)
        return False


def _ensure_model(name):
    """Ensure the requested model is loaded. Swap if needed. Returns True if ready."""
    # Check if correct model is already running (possibly started by another process)
    _, current_model = _read_pid_model()
    if current_model == name and _is_server_alive():
        _local_model = name
        return True
    # Need to swap
    return _start_server(name)


def _acquire_file_lock(timeout=120):
    """Acquire cross-process file lock. Returns lock file handle or None."""
    os.makedirs(os.path.dirname(_LOCK_FILE), exist_ok=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            fh = os.fdopen(fd, "w")
            fh.write(f"{os.getpid()}\n")
            fh.flush()
            return fh
        except FileExistsError:
            # Check if lock is stale (owner dead)
            try:
                with open(_LOCK_FILE) as f:
                    lock_pid = int(f.read().strip())
                # Check if PID is alive
                if platform.system() == "Windows":
                    result = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {lock_pid}"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if str(lock_pid) not in result.stdout:
                        os.remove(_LOCK_FILE)
                        continue
                else:
                    os.kill(lock_pid, 0)  # raises if dead
            except (ProcessLookupError, OSError, ValueError):
                try:
                    os.remove(_LOCK_FILE)
                except Exception:
                    pass
                continue
            time.sleep(1)
    logger.warning("llama-server file lock timeout (%ds)", timeout)
    return None


def _release_file_lock(fh):
    """Release cross-process file lock."""
    if fh is not None:
        try:
            fh.close()
        except Exception:
            pass
        try:
            os.remove(_LOCK_FILE)
        except Exception:
            pass


def query_llama_server(name, prompt, n_predict=1024, temperature=0.0,
                       top_p=0.2, stop=None):
    """Query the shared llama-server. Swaps model if needed.

    Thread-safe and cross-process-safe via file lock.
    Returns completion text or None (caller should fall back to subprocess).
    """
    cfg = _MODEL_CONFIGS.get(name)
    if cfg is None:
        return None

    with _thread_lock:
        fh = _acquire_file_lock(timeout=120)
        if fh is None:
            return None
        try:
            if not _ensure_model(name):
                return None
        finally:
            _release_file_lock(fh)

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
            f"http://127.0.0.1:{_PORT}/completion",
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
    with _thread_lock:
        if name is None or _local_model == name:
            _stop_server()


def stop_all_servers():
    """Stop the llama-server regardless of which model is loaded."""
    with _thread_lock:
        _stop_server()
