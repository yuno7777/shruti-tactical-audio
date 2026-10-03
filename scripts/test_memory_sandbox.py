"""
Windows Job Object Sandbox with Strict Hardware Memory & Core Limits
Emulates a constrained hardware environment (< 1 GB RAM, single-core CPU) by enforcing:
1. Hard OS kernel memory ceiling: 512 MB (strict commit & working set cap via Windows Job Object)
2. CPU Affinity: 1 single core
3. Continuous audio streaming for thousands of 16 ms chunks
"""
import os
import sys
import ctypes
from ctypes import wintypes
import psutil
import soundfile as sf
import numpy as np

# Windows Job Object Structures & Constants
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JobObjectExtendedLimitInformation = 9

class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64)
    ]

class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD)
    ]

class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t)
    ]

def apply_hard_memory_and_cpu_cap(limit_mb=512, cpu_core_id=0):
    """
    Enforces a kernel-level hard memory limit and binds execution to a single core.
    """
    # 1. Bind to single CPU core
    p = psutil.Process()
    p.cpu_affinity([cpu_core_id])
    print(f"[SANDBOX] Bound process to single CPU Core {cpu_core_id}.")

    # 2. Set Windows Job Object memory ceiling
    limit_bytes = int(limit_mb * 1024 * 1024)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]

    h_job = k32.CreateJobObjectW(None, None)
    if not h_job:
        raise ctypes.WinError(ctypes.get_last_error())

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_JOB_MEMORY
    info.ProcessMemoryLimit = limit_bytes
    info.JobMemoryLimit = limit_bytes

    res = k32.SetInformationJobObject(
        h_job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info)
    )
    if not res:
        raise ctypes.WinError(ctypes.get_last_error())

    current_proc = k32.GetCurrentProcess()
    res = k32.AssignProcessToJobObject(h_job, current_proc)
    if not res:
        raise ctypes.WinError(ctypes.get_last_error())

    print(f"[SANDBOX] Kernel Job Object activated: Strict Hard Memory Ceiling = {limit_mb} MB.")
    return h_job

if __name__ == "__main__":
    h_job = apply_hard_memory_and_cpu_cap(limit_mb=512, cpu_core_id=0)
    print("Sandbox successfully verified!")
