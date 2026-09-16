"""Tests for the container lifecycle wrapper.

The point of scripts/service_control.py is that callers stop racing each other:
``start-service`` ends in ``podman run --replace``, so starting the service while
another caller is mid-scan kills that scan once it outlasts podman's SIGTERM
grace period. These tests pin the behaviour that prevents it -- check health
first, hold a lock, only start when genuinely down -- without needing podman or
a GPU.

Run: python3 -m unittest discover -s tests
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import service_control as control  # noqa: E402


HEALTHY = {"ok": True, "cuda_available": True, "gpu_name": "FakeGPU"}
LOADING = {"ok": False, "cuda_available": True}
NO_GPU = {"ok": True, "cuda_available": False}


class StateTest(unittest.TestCase):
    def test_unreachable_is_down(self):
        with mock.patch.object(control, "health", return_value=None):
            self.assertEqual(control.state(), control.STATE_DOWN)

    def test_loading_is_not_ready(self):
        with mock.patch.object(control, "health", return_value=LOADING):
            self.assertEqual(control.state(), control.STATE_NOT_READY)

    def test_missing_gpu_is_its_own_state(self):
        with mock.patch.object(control, "health", return_value=NO_GPU):
            self.assertEqual(control.state(), control.STATE_NO_GPU)

    def test_healthy(self):
        with mock.patch.object(control, "health", return_value=HEALTHY):
            self.assertEqual(control.state(), control.STATE_HEALTHY)

    def test_status_command_maps_every_state_to_an_exit_code(self):
        for state, expected in (
            (control.STATE_HEALTHY, control.EXIT_HEALTHY),
            (control.STATE_DOWN, control.EXIT_DOWN),
            (control.STATE_NOT_READY, control.EXIT_NOT_READY),
            (control.STATE_NO_GPU, control.EXIT_NO_GPU),
        ):
            with mock.patch.object(control, "state", return_value=state):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = control.main(["status"])
                self.assertEqual(code, expected)
                self.assertEqual(out.getvalue().strip(), state)


class EnsureTest(unittest.TestCase):
    def test_healthy_service_is_reused_without_touching_the_start_script(self):
        # The whole point: never --replace a container that is serving.
        with mock.patch.object(control, "state", return_value=control.STATE_HEALTHY), \
                mock.patch.object(control, "run_script") as run_script:
            self.assertEqual(control.ensure(), "reused")
        run_script.assert_not_called()

    def test_down_service_is_started_then_awaited(self):
        states = iter([control.STATE_DOWN, control.STATE_HEALTHY])
        with mock.patch.object(control, "state", side_effect=lambda: next(states)), \
                mock.patch.object(control, "run_script") as run_script:
            self.assertEqual(control.ensure(), "started")
        run_script.assert_called_once()
        self.assertEqual(run_script.call_args.args[0], control.START_SCRIPT)

    def test_container_still_loading_is_waited_out_not_replaced(self):
        # A container that is up but not answering yet may just be loading the
        # model; replacing it would restart that load and kill any in-flight work.
        states = iter([control.STATE_NOT_READY, control.STATE_NOT_READY, control.STATE_HEALTHY])
        with mock.patch.object(control, "state", side_effect=lambda: next(states)), \
                mock.patch.object(control, "container_running", return_value=True), \
                mock.patch.object(control.time, "sleep"), \
                mock.patch.object(control, "run_script") as run_script:
            self.assertEqual(control.ensure(), "reused")
        run_script.assert_not_called()

    def test_port_taken_by_something_unhealthy_is_started_fresh(self):
        # Not ready and no container of ours: nothing to preserve, so start.
        states = iter([control.STATE_NOT_READY, control.STATE_HEALTHY])
        with mock.patch.object(control, "state", side_effect=lambda: next(states)), \
                mock.patch.object(control, "container_running", return_value=False), \
                mock.patch.object(control, "run_script") as run_script:
            self.assertEqual(control.ensure(), "started")
        run_script.assert_called_once()

    def test_service_without_gpu_fails_closed(self):
        with mock.patch.object(control, "state", return_value=control.STATE_NO_GPU), \
                mock.patch.object(control, "run_script") as run_script:
            with self.assertRaises(control.LifecycleError) as ctx:
                control.ensure()
        self.assertEqual(ctx.exception.code, control.EXIT_NO_GPU)
        run_script.assert_not_called()

    def test_never_ready_times_out_with_its_own_code(self):
        with mock.patch.object(control, "state", return_value=control.STATE_DOWN), \
                mock.patch.object(control, "run_script"), \
                mock.patch.object(control.time, "sleep"):
            with self.assertRaises(control.LifecycleError) as ctx:
                control.ensure(timeout=0.01)
        self.assertEqual(ctx.exception.code, control.EXIT_TIMED_OUT)

    def test_failed_start_script_surfaces_its_exit_code(self):
        with mock.patch.object(control, "state", return_value=control.STATE_DOWN), \
                mock.patch.object(
                    control.subprocess, "run",
                    side_effect=subprocess.CalledProcessError(1, control.START_SCRIPT),
                ):
            with self.assertRaises(control.LifecycleError) as ctx:
                control.ensure()
        self.assertEqual(ctx.exception.code, control.EXIT_START_FAILED)


class StopTest(unittest.TestCase):
    def setUp(self):
        self.runtime = tempfile.mkdtemp()
        patcher = mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": self.runtime})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.runtime, True)

    def test_stop_runs_the_stop_script(self):
        with mock.patch.object(control, "container_running", return_value=True), \
                mock.patch.object(control, "active_scans", return_value=0), \
                mock.patch.object(control, "run_script") as run_script:
            self.assertEqual(control.stop(), "stopped")
        self.assertEqual(run_script.call_args.args[0], control.STOP_SCRIPT)

    def test_stop_is_a_no_op_when_nothing_runs(self):
        with mock.patch.object(control, "container_running", return_value=False), \
                mock.patch.object(control, "active_scans", return_value=0), \
                mock.patch.object(control, "run_script") as run_script:
            self.assertEqual(control.stop(), "already-down")
        run_script.assert_not_called()

    def test_stop_refuses_while_a_commit_holds_a_lease(self):
        # "stop when done" is the natural agent habit; it must not kill a commit.
        control.take_lease(os.getpid())
        with mock.patch.object(control, "container_running", return_value=True), \
                mock.patch.object(control, "run_script") as run_script:
            outcome = control.stop()
        self.assertTrue(outcome.startswith("refused"))
        run_script.assert_not_called()

    def test_stop_refuses_while_a_scan_is_running(self):
        with mock.patch.object(control, "active_scans", return_value=2), \
                mock.patch.object(control, "container_running", return_value=True), \
                mock.patch.object(control, "run_script") as run_script:
            outcome = control.stop()
        self.assertTrue(outcome.startswith("refused"))
        run_script.assert_not_called()

    def test_force_overrides_every_guard(self):
        control.take_lease(os.getpid())
        with mock.patch.object(control, "active_scans", return_value=2), \
                mock.patch.object(control, "container_running", return_value=True), \
                mock.patch.object(control, "run_script") as run_script:
            self.assertEqual(control.stop(force=True), "stopped")
        run_script.assert_called_once()
        self.assertEqual(control.live_leases(), [])

    def test_refused_stop_exits_busy(self):
        control.take_lease(os.getpid())
        out = io.StringIO()
        with mock.patch.object(control, "container_running", return_value=True), \
                mock.patch.object(control, "run_script"), \
                contextlib.redirect_stdout(out):
            code = control.main(["stop"])
        self.assertEqual(code, control.EXIT_BUSY)


class LockTest(unittest.TestCase):
    def test_two_ensures_serialize_and_only_one_starts(self):
        # Two commits landing together must not both run --replace.
        starts = []
        gate = threading.Event()
        healthy_after_start = threading.Event()

        def fake_state():
            return control.STATE_HEALTHY if healthy_after_start.is_set() else control.STATE_DOWN

        def fake_run_script(script, action):
            starts.append(script)
            gate.wait(timeout=5)       # hold the lock while "starting"
            healthy_after_start.set()

        results = []
        with mock.patch.object(control, "state", side_effect=fake_state), \
                mock.patch.object(control, "container_running", return_value=False), \
                mock.patch.object(control, "run_script", side_effect=fake_run_script):
            threads = [
                threading.Thread(target=lambda: results.append(control.ensure()))
                for _ in range(2)
            ]
            threads[0].start()
            time.sleep(0.2)            # let the first thread take the lock
            threads[1].start()
            gate.set()
            for thread in threads:
                thread.join(timeout=10)

        self.assertEqual(len(starts), 1, "both callers started the container")
        self.assertEqual(sorted(results), ["reused", "started"])

    def test_a_held_lock_reports_busy_rather_than_starting(self):
        holder = control.Lock()
        holder.__enter__()
        try:
            with mock.patch.object(control, "run_script") as run_script:
                with self.assertRaises(control.LifecycleError) as ctx:
                    with control.Lock(timeout=0.2):
                        pass
            self.assertEqual(ctx.exception.code, control.EXIT_BUSY)
            run_script.assert_not_called()
        finally:
            holder.__exit__()


class LeaseTest(unittest.TestCase):
    """Teardown is refcounted, so a fast commit cannot stop a slow one's service.

    Reproduced before this existed: commit A started the service, finished its
    tiny scan and stopped the container while commit B was still scanning a
    large diff; B failed with "not reachable".
    """

    def setUp(self):
        self.runtime = tempfile.mkdtemp()
        patcher = mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": self.runtime})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.runtime, True)

    def test_last_lease_out_stops_the_service(self):
        with mock.patch.object(control, "state", return_value=control.STATE_DOWN), \
                mock.patch.object(control, "wait_until_ready"), \
                mock.patch.object(control, "run_script"):
            control.ensure(lease=True)
        self.assertTrue(control.autostart_marker().exists())

        with mock.patch.object(control, "container_running", return_value=True), \
                mock.patch.object(control, "active_scans", return_value=0), \
                mock.patch.object(control, "run_script") as run_script:
            self.assertEqual(control.release(), "stopped")
        self.assertEqual(run_script.call_args.args[0], control.STOP_SCRIPT)
        self.assertFalse(control.autostart_marker().exists())

    def test_service_stays_up_while_another_lease_is_open(self):
        with mock.patch.object(control, "state", return_value=control.STATE_DOWN), \
                mock.patch.object(control, "wait_until_ready"), \
                mock.patch.object(control, "run_script"):
            control.ensure(lease=True)
        control.take_lease(os.getpid() + 100_000)   # a second, live-looking caller

        with mock.patch.object(control, "pid_alive", return_value=True), \
                mock.patch.object(control, "run_script") as run_script:
            outcome = control.release()
        self.assertIn("held by 1", outcome)
        run_script.assert_not_called()

    def test_a_running_scan_defers_teardown(self):
        # An agent scanning by hand holds no lease; the in-flight count covers it.
        with mock.patch.object(control, "state", return_value=control.STATE_DOWN), \
                mock.patch.object(control, "wait_until_ready"), \
                mock.patch.object(control, "run_script"):
            control.ensure(lease=True)

        with mock.patch.object(control, "active_scans", return_value=1), \
                mock.patch.object(control, "run_script") as run_script:
            self.assertEqual(control.release(), "held by a running scan")
        run_script.assert_not_called()

    def test_a_hand_started_service_is_never_stopped_by_release(self):
        # ensure() without --lease leaves no marker, so it is not ours to stop.
        with mock.patch.object(control, "state", return_value=control.STATE_DOWN), \
                mock.patch.object(control, "wait_until_ready"), \
                mock.patch.object(control, "run_script"):
            control.ensure()

        with mock.patch.object(control, "run_script") as run_script:
            self.assertEqual(control.release(), "left-running")
        run_script.assert_not_called()

    def test_dead_holders_do_not_pin_the_gpu_forever(self):
        # A commit killed with SIGKILL cannot drop its lease.
        control.take_lease(999_999_999)
        with mock.patch.object(control, "state", return_value=control.STATE_DOWN), \
                mock.patch.object(control, "wait_until_ready"), \
                mock.patch.object(control, "run_script"):
            control.ensure(lease=True)

        with mock.patch.object(control, "container_running", return_value=True), \
                mock.patch.object(control, "active_scans", return_value=0), \
                mock.patch.object(control, "run_script") as run_script:
            self.assertEqual(control.release(), "stopped")
        run_script.assert_called_once()

    def test_forced_stop_clears_leases_and_marker(self):
        # The escape hatch has to leave clean state behind, or the next ensure
        # would inherit a lease nobody holds.
        control.take_lease(os.getpid())
        control.autostart_marker().touch()
        with mock.patch.object(control, "container_running", return_value=True), \
                mock.patch.object(control, "run_script"):
            self.assertEqual(control.stop(force=True), "stopped")
        self.assertEqual(control.live_leases(), [])
        self.assertFalse(control.autostart_marker().exists())

    def test_active_scans_defaults_to_zero_when_unavailable(self):
        with mock.patch.object(control, "health", return_value=None):
            self.assertEqual(control.active_scans(), 0)
        with mock.patch.object(control, "health", return_value={"active_scans": "?"}):
            self.assertEqual(control.active_scans(), 0)
        with mock.patch.object(control, "health", return_value={"active_scans": 3}):
            self.assertEqual(control.active_scans(), 3)


class HealthTest(unittest.TestCase):
    def test_unreachable_service_reports_none(self):
        with mock.patch.object(
            control, "request_json", side_effect=control.ServiceUnreachable("down")
        ):
            self.assertIsNone(control.health())

    def test_http_error_is_not_mistaken_for_down(self):
        # The port answered, so the service exists; it is just not healthy.
        with mock.patch.object(
            control, "request_json", side_effect=control.ServiceError(503, "no_gpu", "nope")
        ):
            self.assertEqual(control.health(), {})
            self.assertEqual(control.state(), control.STATE_NOT_READY)


if __name__ == "__main__":
    unittest.main()
