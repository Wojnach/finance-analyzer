"""Subprocess utilities to prevent orphaned child processes on Windows.

Provides:
- run_safe(): Drop-in subprocess.run() replacement that uses Windows Job Objects
  with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE so children die when the parent exits.
- popen_in_job(): Popen wrapper for long-running subprocesses — assigns the child
  to a Job Object so it's automatically killed if the parent dies.
- kill_orphaned_by_cmdline(): Find and kill orphaned processes matching a command
  line pattern (safety net for processes that escaped Job Object protection).
- kill_orphaned_llama(): Safety-net reaper for orphaned llama-completion.exe processes.
"""

import json
import logging
import subprocess
import sys

logger = logging.getLogger("portfolio.subprocess_utils")


def run_safe(cmd, **kwargs):
    """Run a subprocess with Windows Job Object protection.

    Drop-in replacement for subprocess.run().  On Windows, creates a Job Object
    with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE so that the child process is
    automatically killed if the parent Python process dies.

    Falls back to plain subprocess.run() on non-Windows or if Job Object
    creation fails.

    Supported kwargs: capture_output, text, timeout, input, stdin (and any
    others accepted by subprocess.Popen / subprocess.run).
    """
    if sys.platform != "win32":
        return subprocess.run(cmd, **kwargs)

    try:
        return _run_with_job_object(cmd, **kwargs)
    except Exception as exc:
        logger.debug("Job Object creation failed (%s), falling back to subprocess.run", exc)
        return subprocess.run(cmd, **kwargs)


def _create_job_object():
    """Create a Windows Job Object with KILL_ON_JOB_CLOSE.

    Returns (job_handle, kernel32) or raises OSError.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError("CreateJobObjectW failed")

    # JOBOBJECT_BASIC_LIMIT_INFORMATION (64-bit layout)
    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("_pad0", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("_pad1", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JobObjectExtendedLimitInformation = 9

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

    ok = kernel32.SetInformationJobObject(
        job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        kernel32.CloseHandle(job)
        raise OSError("SetInformationJobObject failed")

    return job, kernel32


def _run_with_job_object(cmd, **kwargs):
    """Internal: run subprocess inside a Windows Job Object."""
    job, kernel32 = _create_job_object()

    try:
        popen_kwargs = dict(kwargs)
        timeout = popen_kwargs.pop("timeout", None)

        if popen_kwargs.pop("capture_output", False):
            popen_kwargs["stdout"] = subprocess.PIPE
            popen_kwargs["stderr"] = subprocess.PIPE

        input_data = popen_kwargs.pop("input", None)
        if input_data is not None and "stdin" not in popen_kwargs:
            popen_kwargs["stdin"] = subprocess.PIPE

        proc = subprocess.Popen(cmd, **popen_kwargs)

        try:
            kernel32.AssignProcessToJobObject(job, int(proc._handle))
        except Exception as e:
            logger.warning(
                "Job Object assignment failed for pid %d — child may orphan: %s",
                proc.pid, e,
            )

        try:
            stdout, stderr = proc.communicate(input=input_data, timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise

        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )
    finally:
        kernel32.CloseHandle(job)


def popen_in_job(cmd, **kwargs):
    """Start a long-running subprocess inside a Windows Job Object.

    Like subprocess.Popen(), but assigns the child to a Job Object with
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. If the parent process dies (crash,
    kill -9, power loss), the OS automatically kills the child.

    Returns (proc, job_handle) on Windows. On non-Windows or if Job Object
    creation fails, returns (proc, None).

    Caller must call close_job(job_handle) when explicitly stopping the child.
    """
    proc = subprocess.Popen(cmd, **kwargs)

    if sys.platform != "win32":
        return proc, None

    try:
        job, kernel32 = _create_job_object()
        kernel32.AssignProcessToJobObject(job, int(proc._handle))
        return proc, job
    except Exception as exc:
        logger.debug("Job Object creation failed for Popen (%s), no auto-cleanup", exc)
        return proc, None


def close_job(job_handle):
    """Close a Job Object handle.

    Safe to call after the child has already been terminated — closing the
    handle on a dead process is a no-op. Call this in your explicit stop
    function after terminating the child.
    """
    if job_handle is None:
        return
    try:
        import ctypes
        ctypes.windll.kernel32.CloseHandle(job_handle)
    except Exception:
        pass


def kill_orphaned_by_cmdline(pattern, exclude_pid=None):
    """Find and kill processes whose command line contains *pattern*.

    Used at startup to sweep orphaned subprocesses from a previous crash.
    Skips the current process and *exclude_pid* if given.

    Returns the number of processes killed. Returns 0 on non-Windows.
    """
    if sys.platform != "win32":
        return 0

    my_pid = __import__("os").getpid()
    skip = {my_pid}
    if exclude_pid is not None:
        skip.add(exclude_pid)

    try:
        result = subprocess.run(
            ["wmic", "process", "where",
             f"CommandLine like '%{pattern}%'",
             "get", "ProcessId", "/format:csv"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception as exc:
        logger.debug("WMIC process query failed: %s", exc)
        return 0

    killed = 0
    for line in result.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[-1])
        except ValueError:
            continue
        if pid in skip or pid == 0:
            continue

        logger.info("Killing orphaned process (pattern=%r): PID %d", pattern, pid)
        try:
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True, timeout=10,
            )
            killed += 1
        except Exception:
            pass

    return killed


def kill_orphaned_llama():
    """Find and kill orphaned llama-completion.exe processes.

    An orphaned process is one whose parent PID no longer exists.
    Uses PowerShell + Win32 API to enumerate and check processes.

    Returns the number of processes killed.  Returns 0 on non-Windows.
    """
    if sys.platform != "win32":
        return 0

    import ctypes

    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    PROCESS_TERMINATE = 0x0001

    # --- Get llama-completion.exe processes via PowerShell --------------------
    ps_cmd = (
        'powershell.exe -NoProfile -Command "'
        "Get-CimInstance Win32_Process -Filter \\\"Name='llama-completion.exe'\\\" "
        '| Select-Object ProcessId,ParentProcessId | ConvertTo-Json"'
    )

    try:
        result = subprocess.run(
            ps_cmd,
            capture_output=True,
            text=True,
            timeout=15,
            shell=True,
        )
    except Exception as exc:
        logger.debug("PowerShell process query failed: %s", exc)
        return 0

    if result.returncode != 0 or not result.stdout.strip():
        return 0

    try:
        data = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        logger.debug("Failed to parse PowerShell JSON output")
        return 0

    # PowerShell returns a single object (not array) when there's only one match
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return 0

    killed = 0
    for entry in data:
        pid = entry.get("ProcessId")
        ppid = entry.get("ParentProcessId")
        if pid is None or ppid is None:
            continue

        # Check if parent is alive
        parent_alive = False
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(ppid))
        if handle:
            parent_alive = True
            kernel32.CloseHandle(handle)

        if not parent_alive:
            logger.info("Orphaned llama-completion.exe PID %d (parent %d dead) — killing", pid, ppid)
            # Terminate the orphan
            h_proc = kernel32.OpenProcess(PROCESS_TERMINATE, False, int(pid))
            if h_proc:
                kernel32.TerminateProcess(h_proc, 1)
                kernel32.CloseHandle(h_proc)
                killed += 1
                logger.info("Killed orphaned llama-completion.exe PID %d", pid)
            else:
                logger.warning("Could not open llama-completion.exe PID %d for termination", pid)

    return killed
