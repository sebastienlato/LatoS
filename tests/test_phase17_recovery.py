"""Administration-only regression checks; no learned models or optimizer updates."""

import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("recover.py")
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parents[1] / "experiments/phase-17/recovery/recover.py"
spec = importlib.util.spec_from_file_location("phase17_recovery_admin", SCRIPT)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_append_journal_never_replaces_ledger(self):
        ledger = self.root / "ledger.json"
        ledger.write_text('{"status":"supervisor-failure"}')
        before = ledger.read_bytes()
        journal = recovery.Journal(self.root / "new.events.jsonl")
        with patch.object(Path, "replace", side_effect=PermissionError("sharing violation")):
            journal.add({"event": "start"})
            with ledger.open("r"):
                journal.add({"event": "complete"})
        journal.close()
        self.assertEqual(ledger.read_bytes(), before)
        self.assertEqual(len((self.root / "new.events.jsonl").read_text().splitlines()), 2)
        with self.assertRaises(FileExistsError):
            recovery.Journal(self.root / "new.events.jsonl")

    def test_original_file_corruption_rejected(self):
        path = self.root / "state.json"
        path.write_bytes(b"original")
        records = {"state.json": {"bytes": 8, "sha256": recovery.digest(path)}}
        recovery.verify_records(self.root, records)
        path.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            recovery.verify_records(self.root, records)

    def test_unsafe_plan_paths_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            recovery.verify_records(self.root, {"../outside": {"bytes": 0, "sha256": "0" * 64}})

    def test_lock_refuses_overlap_without_changing_bytes(self):
        path = self.root / "active.lock"
        path.write_bytes(b"0")
        expected = {"active.lock": {"bytes": 1, "sha256": recovery.digest(path)}}
        with recovery.lock(path) as lease:
            recovery.verify_records(self.root, expected, {"active.lock": lease})
            with self.assertRaises(OSError), recovery.lock(path):
                self.fail("Concurrent controller acquired lock")
        self.assertEqual(path.read_bytes(), b"0")

    def test_hard_timeout_stops_child_and_preserves_journal(self):
        journal = recovery.Journal(self.root / "timeout.events.jsonl")
        with self.assertRaises(TimeoutError):
            recovery.run_child(
                [sys.executable, "-c", "import time;time.sleep(60)"],
                self.root,
                self.root / "child.log",
                journal,
                time.monotonic(),
                0.1,
                lambda: 0,
                100,
            )
        journal.close()
        records = [
            json.loads(s) for s in (self.root / "timeout.events.jsonl").read_text().splitlines()
        ]
        self.assertTrue(any(r["event"] == "worker-start" for r in records))
        self.assertFalse(any(r["event"] == "worker-complete" for r in records))

    def test_failed_child_not_success(self):
        journal = recovery.Journal(self.root / "failed.events.jsonl")
        with self.assertRaisesRegex(RuntimeError, "exit code"):
            recovery.run_child(
                [sys.executable, "-c", "raise SystemExit(9)"],
                self.root,
                self.root / "child.log",
                journal,
                time.monotonic(),
                30,
                lambda: 0,
                100,
            )
        journal.close()

    def test_receipt_without_terminal_event_or_with_failure_rejected(self):
        result = {"status": "complete", "resumptions_used": 1}
        recovery.write_once(self.root / "recover-result.json", result)
        journal = recovery.Journal(self.root / "recover.events.jsonl")
        journal.add({"event": "worker-complete"})
        with self.assertRaisesRegex(ValueError, "complete journal"):
            recovery.verified_receipt(self.root, "recover")
        journal.add({"event": "complete", "result": result})
        journal.close()
        self.assertEqual(recovery.verified_receipt(self.root, "recover"), result)
        recovery.write_once(self.root / "recover-failure.json", {"error": "late disk failure"})
        with self.assertRaisesRegex(ValueError, "failure record"):
            recovery.verified_receipt(self.root, "recover")

    def test_replay_counters_and_loss_are_checked(self):
        for name in ("segment-0", "segment-1"):
            (self.root / name).mkdir()
        rows = [
            {
                "step": i,
                "targets": 2,
                "tokens_seen": 2 * i,
                "windows_seen": i,
                "epoch": 0,
                "microbatches": 2 if i == 7485 else 8,
                "loss": 1.0,
                "learning_rate": 0.00003,
                "grad_norm_before_clip": 1.0,
            }
            for i in range(1, 7486)
        ]
        for name, subset in [("segment-0", rows), ("segment-1", rows[7000:])]:
            (self.root / name / "updates.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in subset)
            )
        recovery.write_once(
            self.root / "segment-1/start.json",
            {
                "start_step": 7000,
                "tokens_seen": 14000,
                "windows_seen": 112000,
                "previous_segment_logged_targets_rolled_back": 970,
            },
        )
        plan = {
            "checkpoint_step": 7000,
            "retained_targets": 14000,
            "replay_targets": 970,
            "final_targets": 14970,
        }
        self.assertEqual(recovery.tail_metrics(self.root, plan)["replayed_updates"], 485)
        rows[-1]["loss"] = 1.01
        (self.root / "segment-1/updates.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows[7000:])
        )
        with self.assertRaisesRegex(ValueError, "recovery loss tolerance"):
            recovery.tail_metrics(self.root, plan)

    @unittest.skipUnless(sys.platform == "win32", "Actual Windows file-sharing API required")
    def test_windows_read_handle_blocks_replace_but_not_separate_journal(self):
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        ledger = self.root / "ledger.json"
        ledger.write_bytes(b"original")
        temporary = self.root / "ledger.json.tmp"
        temporary.write_bytes(b"new")
        # Read access and FILE_SHARE_READ only: rename/delete sharing is absent.
        handle = create(str(ledger), 0x80000000, 1, None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            with self.assertRaises(PermissionError):
                temporary.replace(ledger)
            journal = recovery.Journal(self.root / "recovery.events.jsonl")
            journal.add({"event": "complete"})
            journal.close()
            self.assertEqual(ledger.read_bytes(), b"original")
        finally:
            kernel.CloseHandle(handle)


if __name__ == "__main__":
    unittest.main()
