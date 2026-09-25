"""Standalone Windows guardian: starts the only study clock before bootstrap/extraction.

A kill-on-close Windows Job Object contains bootstrap, controller and every child.
No torch import, shell, environment install, training or archive preparation precedes
that clock. Keep this file beside the transferred ZIP; do not extract it yourself.
"""

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

STARTED = time.monotonic()


def windows_types():
    import ctypes as c
    from ctypes import wintypes as w

    class Basic(c.Structure):
        _fields_ = [
            ("ProcessTime", c.c_int64),
            ("JobTime", c.c_int64),
            ("Flags", w.DWORD),
            ("MinWS", c.c_size_t),
            ("MaxWS", c.c_size_t),
            ("Processes", w.DWORD),
            ("Affinity", c.c_size_t),
            ("Priority", w.DWORD),
            ("Scheduling", w.DWORD),
        ]

    class IO(c.Structure):
        _fields_ = [
            (name, c.c_uint64)
            for name in ("ReadOps", "WriteOps", "OtherOps", "Read", "Write", "Other")
        ]

    class Limits(c.Structure):
        _fields_ = [
            ("Basic", Basic),
            ("IO", IO),
            ("ProcessMem", c.c_size_t),
            ("JobMem", c.c_size_t),
            ("PeakProcessMem", c.c_size_t),
            ("PeakJobMem", c.c_size_t),
        ]

    class Startup(c.Structure):
        _fields_ = [
            ("cb", w.DWORD),
            ("reserved", w.LPWSTR),
            ("desktop", w.LPWSTR),
            ("title", w.LPWSTR),
            ("x", w.DWORD),
            ("y", w.DWORD),
            ("sx", w.DWORD),
            ("sy", w.DWORD),
            ("xc", w.DWORD),
            ("yc", w.DWORD),
            ("fill", w.DWORD),
            ("flags", w.DWORD),
            ("show", w.WORD),
            ("reserved2", w.WORD),
            ("bytes", c.POINTER(w.BYTE)),
            ("stdin", w.HANDLE),
            ("stdout", w.HANDLE),
            ("stderr", w.HANDLE),
        ]

    class Process(c.Structure):
        _fields_ = [("process", w.HANDLE), ("thread", w.HANDLE), ("pid", w.DWORD), ("tid", w.DWORD)]

    return Limits, Startup, Process


def guarded(command, hard_end):
    import ctypes as c
    from ctypes import wintypes as w

    Limits, Startup, Process = windows_types()
    k = c.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([c.c_void_p, w.LPCWSTR], w.HANDLE),
        "SetInformationJobObject": ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
        "CreateProcessW": (
            [
                w.LPCWSTR,
                w.LPWSTR,
                c.c_void_p,
                c.c_void_p,
                w.BOOL,
                w.DWORD,
                c.c_void_p,
                w.LPCWSTR,
                c.POINTER(Startup),
                c.POINTER(Process),
            ],
            w.BOOL,
        ),
        "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
        "ResumeThread": ([w.HANDLE], w.DWORD),
        "WaitForSingleObject": ([w.HANDLE, w.DWORD], w.DWORD),
        "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
        "TerminateProcess": ([w.HANDLE, w.UINT], w.BOOL),
        "GetExitCodeProcess": ([w.HANDLE, c.POINTER(w.DWORD)], w.BOOL),
        "CloseHandle": ([w.HANDLE], w.BOOL),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(k, name)
        fn.argtypes, fn.restype = args, result

    def check(ok):
        if not ok:
            raise c.WinError(c.get_last_error())

    job = k.CreateJobObjectW(None, None)
    check(job)
    pi, si = Process(), Startup()
    si.cb = c.sizeof(si)
    assigned = False
    try:
        limits = Limits()
        limits.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway.
        check(k.SetInformationJobObject(job, 9, c.byref(limits), c.sizeof(limits)))
        check(
            k.CreateProcessW(
                None,
                c.create_unicode_buffer(subprocess.list2cmdline(command)),
                None,
                None,
                False,
                4,
                None,
                None,
                c.byref(si),
                c.byref(pi),
            )
        )
        # Suspended creation prevents work/descendants escaping before assignment.
        check(k.AssignProcessToJobObject(job, pi.process))
        assigned = True
        if time.monotonic() >= hard_end:
            raise TimeoutError("Deadline exhausted before dispatch")
        require_resume = k.ResumeThread(pi.thread)
        if require_resume == 0xFFFFFFFF:
            raise c.WinError(c.get_last_error())
        remaining = max(0, int((hard_end - time.monotonic()) * 1000))
        wait = k.WaitForSingleObject(pi.process, remaining)
        if wait != 0:
            check(k.TerminateJobObject(job, 124))
            k.WaitForSingleObject(pi.process, 1000)
            return {"exit_code": 124, "deadline_terminated": True, "wait_result": wait}
        code = w.DWORD()
        check(k.GetExitCodeProcess(pi.process, c.byref(code)))
        return {"exit_code": code.value, "deadline_terminated": False}
    finally:
        # Covers timeout, interruption, API failure and any surviving descendants.
        if assigned:
            k.TerminateJobObject(job, 125)
        elif pi.process:
            k.TerminateProcess(pi.process, 125)
        for handle in (pi.thread, pi.process, job):
            if handle:
                k.CloseHandle(handle)


def guard_selftest(root):
    """Actual Windows nested-job timeout, including a grandchild; preparation time only."""
    import ctypes as c
    from ctypes import wintypes as w

    pidfile = root / "guardian-test-pids.json"
    program = (
        "import subprocess,sys,time,json,os; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']); "
        "open(sys.argv[1],'w').write(json.dumps([os.getpid(),p.pid]));time.sleep(120)"
    )
    began = time.monotonic()
    result = guarded([sys.executable, "-c", program, str(pidfile)], began + 2)
    if not result["deadline_terminated"] or not pidfile.exists():
        raise ValueError("Guardian process-tree fixture did not run/terminate")
    k = c.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.argtypes, k.OpenProcess.restype = [w.DWORD, w.BOOL, w.DWORD], w.HANDLE
    k.WaitForSingleObject.argtypes, k.WaitForSingleObject.restype = [w.HANDLE, w.DWORD], w.DWORD
    k.CloseHandle.argtypes = [w.HANDLE]
    for pid in json.loads(pidfile.read_text()):
        handle = k.OpenProcess(0x00100000, False, pid)
        if handle:
            try:
                if k.WaitForSingleObject(handle, 1000) != 0:
                    raise ValueError("Guardian left a process alive")
            finally:
                k.CloseHandle(handle)
        elif c.get_last_error() != 87:
            raise c.WinError(c.get_last_error())
    return {
        "passed": True,
        "elapsed_seconds": time.monotonic() - began,
        "optimizer_updates": 0,
        "scope": "Windows nested-job descendant kill",
    }


def bootstrap(args):
    import zipfile

    root = args.transfer.resolve() / "subset-confirmation-v1"
    with (root / "bootstrap.log").open("x", encoding="utf-8") as log:
        sys.stdout = sys.stderr = log
        try:
            with args.archive.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            if digest != args.sha256:
                raise ValueError("Transfer ZIP checksum mismatch")
            launch_record = json.loads((root / "launch.json").read_text())
            token = os.environ.pop("LATOS_SUBSET_GUARD_TOKEN", "")
            if (
                not token
                or hashlib.sha256(token.encode()).hexdigest() != launch_record["ticket"]
                or args.started != launch_record["started_monotonic"]
            ):
                raise ValueError("Bootstrap launch binding differs")
            with zipfile.ZipFile(args.archive) as z:
                manifest = json.loads(z.read("bundle.json"))
                if (
                    hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                    != manifest["files"]["launch.py"]["sha256"]
                ):
                    raise ValueError("Standalone guardian differs from reviewed bundle")
            destination = root / "bootstrap"
            destination.mkdir()
            with zipfile.ZipFile(args.archive) as z:
                for name in ("cf_common.py", "cf_controller.py", "cf_evidence.py"):
                    (destination / name).write_bytes(z.read(name))
            sys.path.insert(0, str(destination))
            from cf_controller import run

            os.environ["LATOS_SUBSET_JOB_ACTIVE"] = "1"
            args.output = root / "run"
            run(args)
        except BaseException:
            import traceback

            traceback.print_exc()
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transfer", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--origin", type=float, help="Original Windows QPC seconds from kickoff")
    p.add_argument("--bootstrap", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--started", type=float, help=argparse.SUPPRESS)
    args = p.parse_args()
    if os.name != "nt":
        raise SystemExit("Windows guardian only; Mac execution prohibited")
    if args.bootstrap:
        # A direct bootstrap is not an alternate unguarded entry point.
        import ctypes as c
        from ctypes import wintypes as w

        k = c.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = w.HANDLE
        k.IsProcessInJob.argtypes = [w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)]
        k.IsProcessInJob.restype = w.BOOL
        present = w.BOOL()
        if not k.IsProcessInJob(k.GetCurrentProcess(), None, c.byref(present)) or not present.value:
            raise SystemExit("Bootstrap requires the guardian job")
        if args.started is None or not 0 <= time.monotonic() - args.started < 180:
            raise SystemExit("Missing original study clock")
        bootstrap(args)
        return
    if "QueryPerformanceCounter" not in time.get_clock_info("monotonic").implementation:
        raise SystemExit("Unrecognized Windows clock; do not execute")
    if args.origin is None or not 0 <= STARTED - args.origin < 180:
        raise SystemExit("Use the kickoff clock; missing, future or expired origin")
    began = args.origin
    root = args.transfer.resolve() / "subset-confirmation-v1"
    root.mkdir(exist_ok=False)  # Permanent one-attempt claim; never delete to retry.
    token = secrets.token_hex(32)
    os.environ["LATOS_SUBSET_GUARD_TOKEN"] = token
    (root / "launch.json").write_text(
        json.dumps(
            {
                "started_monotonic": began,
                "archive": str(args.archive.resolve()),
                "expected_sha256": args.sha256,
                "ticket": hashlib.sha256(token.encode()).hexdigest(),
                "hard_seconds": 1800,
                "guardian_kill_seconds": 1795,
                "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--bootstrap",
        "--transfer",
        str(args.transfer.resolve()),
        "--archive",
        str(args.archive.resolve()),
        "--sha256",
        args.sha256,
        "--started",
        str(began),
    ]
    try:
        result = guarded(command, began + 1795)
    except BaseException as exc:
        result = {"exit_code": 125, "error": str(exc)}
    result["total_wall_seconds"] = time.monotonic() - began
    (root / "guardian.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
