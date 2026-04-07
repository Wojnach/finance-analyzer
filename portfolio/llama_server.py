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
from contextlib import suppress

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


def _kill_by_port():
    """Kill any process listening on _PORT (catches orphaned servers)."""
    try:
        if platform.system() == "Windows":
            # Find all PIDs on our port
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True, timeout=10,
            )
            pids_to_kill = set()
            for line in result.stdout.splitlines():
                if f":{_PORT}" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        with suppress(ValueError):
                            pids_to_kill.add(int(parts[-1]))
            for pid in pids_to_kill:
                if pid != os.getpid():
                    logger.info("Killing orphaned process on port %d: PID %d", _PORT, pid)
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        capture_output=True, timeout=10,
                    )
        else:
            result = subprocess.run(
                ["fuser", f"{_PORT}/tcp"],
                capture_output=True, text=True, timeout=10,
            )
            for pid_str in result.stdout.split():
                with suppress(ValueError):
                    pid = int(pid_str)
                    if pid != os.getpid():
                        os.kill(pid, 9)
    except Exception as e:
        logger.debug("Port kill check failed: %s", e)


def _is_llama_server_process(pid):
    """Verify a PID is actually a llama-server process before killing it."""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "process", "where", f"ProcessId={pid}", "get", "Name"],
                capture_output=True, text=True, timeout=5,
            )
            return "llama-server" in result.stdout.lower()
        else:
            with open(f"/proc/{pid}/cmdline") as f:
                return "llama-server" in f.read()
    except Exception as e:
        logger.warning("Llama process check failed for PID %s: %s", pid, e, exc_info=True)
        return False


def _kill_server_by_pid():
    """Kill any existing llama-server via saved PID file.

    Validates the process is actually llama-server before killing to
    prevent PID-reuse collateral damage (Codex finding #2).
    """
    try:
        if os.path.exists(_PID_FILE):
            with open(_PID_FILE) as f:
                content = f.read().strip()
            if content:
                pid = int(content.split(":")[0])
                if not _is_llama_server_process(pid):
                    logger.warning("PID %d from pid file is not llama-server, skipping kill", pid)
                    return
                if platform.system() == "Windows":
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        capture_output=True, timeout=10,
                    )
                else:
                    os.kill(pid, 9)
                time.sleep(1)
    except Exception:
        logger.debug("Failed to kill server pid=%s", pid)


def _write_pid(proc, model_name):
    """Write server PID + model name so other processes know what's loaded."""
    try:
        os.makedirs(os.path.dirname(_PID_FILE), exist_ok=True)
        with open(_PID_FILE, "w") as f:
            f.write(f"{proc.pid}:{model_name}")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        logger.warning("PID file write failed: %s", e)


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
        logger.debug("Failed to read PID file %s", _PID_FILE)
    return None, None


def _is_server_alive():
    """Check if llama-server is responding on the port."""
    try:
        r = _requests.get(f"http://127.0.0.1:{_PORT}/health", timeout=2)
        return r.status_code == 200
    except Exception as e:
        logger.warning("Llama health check failed: %s", e, exc_info=True)
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
            with suppress(Exception):
                _local_proc.kill()
    _local_proc = None
    _local_model = None

    # Safety net: kill anything still on port 8787 (catches orphaned servers
    # not tracked by PID file — the root cause of 10x process accumulation)
    _kill_by_port()

    with suppress(OSError):
        os.remove(_PID_FILE)


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
    # Wait for GPU driver to reclaim VRAM after killing the previous server.
    # Windows VRAM release is asynchronous — 1s was insufficient, causing
    # Qwen3 to see 136MB free instead of ~5GB after Ministral teardown.
    time.sleep(4)

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
        return True
    # Need to swap
    return _start_server(name)


def _acquire_file_lock(timeout=300):
    """Acquire cross-process file lock. Returns lock file handle or None.

    Timeout must exceed the HTTP query timeout (240s) to prevent callers
    from falling back to subprocess while the server is still handling a
    legitimate query (Codex review finding #1).
    """
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
                with suppress(OSError):
                    os.remove(_LOCK_FILE)
                continue
            time.sleep(1)
    logger.warning("llama-server file lock timeout (%ds)", timeout)
    return None


def _release_file_lock(fh):
    """Release cross-process file lock."""
    if fh is not None:
        with suppress(Exception):
            fh.close()
        with suppress(OSError):
            os.remove(_LOCK_FILE)


def query_llama_server(name, prompt, n_predict=1024, temperature=0.0,
                       top_p=0.2, stop=None):
    """Query the shared llama-server. Swaps model if needed.

    Thread-safe and cross-process-safe via file lock.
    Returns completion text or None (caller should fall back to subprocess).
    """
    cfg = _MODEL_CONFIGS.get(name)
    if cfg is None:
        return None

    # BUG-165: Hold both locks for the entire model-swap + query operation.
    # Releasing locks between swap and query allowed another thread/process to
    # swap the model mid-query, killing the server and causing silent failures.
    # Serialization is correct here — only one 8B model fits in VRAM at a time.
    with _thread_lock:
        fh = _acquire_file_lock(timeout=300)
        if fh is None:
            return None
        try:
            if not _ensure_model(name):
                return None
            text = _query_http(prompt, n_predict, temperature, top_p, stop)
            if text is None:
                logger.warning("llama-server %s returned empty response", name)
            return text
        except Exception as e:
            logger.warning("llama-server %s query failed: %s", name, e)
            return None
        finally:
            _release_file_lock(fh)


def _query_http(prompt, n_predict=1024, temperature=0.0, top_p=0.2, stop=None):
    """Send an HTTP completion request. No locking — caller must hold locks."""
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
    return None


def query_llama_server_batch(name, prompts_and_params):
    """Query the server for multiple prompts, holding the lock for the entire batch.

    Prevents other processes from swapping the model between items (Codex finding #4).

    Args:
        name: model name (e.g. "ministral3", "qwen3")
        prompts_and_params: list of dicts with keys: prompt, n_predict, temperature, top_p, stop

    Returns:
        list of (completion_text_or_None) in same order as input.
    """
    cfg = _MODEL_CONFIGS.get(name)
    if cfg is None:
        return [None] * len(prompts_and_params)

    results = []
    with _thread_lock:
        fh = _acquire_file_lock(timeout=300)
        if fh is None:
            return [None] * len(prompts_and_params)
        try:
            if not _ensure_model(name):
                return [None] * len(prompts_and_params)
            for params in prompts_and_params:
                try:
                    text = _query_http(
                        params["prompt"],
                        n_predict=params.get("n_predict", 1024),
                        temperature=params.get("temperature", 0.0),
                        top_p=params.get("top_p", 0.2),
                        stop=params.get("stop"),
                    )
                    results.append(text)
                except Exception as e:
                    logger.warning("llama-server batch query failed: %s", e)
                    results.append(None)
        finally:
            _release_file_lock(fh)
    return results


def stop_server(name=None):
    """Stop the llama-server (optionally only if a specific model is loaded)."""
    with _thread_lock:
        if name is None or _local_model == name:
            _stop_server()


def stop_all_servers():
    """Stop the llama-server regardless of which model is loaded."""
    with _thread_lock:
        _stop_server()
