"""Regression tests for how the pre-commit hook reports service errors.

Guards the issue-2 fix: an HTTP 500/503 from a reachable service must surface its
status and body, not be misreported as "service is not reachable" (which sends
people restarting an already-running service). HTTPError is a subclass of
URLError, so order of the except clauses matters.

Run: python3 -m unittest discover -s tests
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import git_pre_commit_pii as hook  # noqa: E402


def make_http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://127.0.0.1:8757/check",
        code=code,
        msg="Service Unavailable",
        hdrs=None,
        fp=io.BytesIO(body.encode("utf-8")),
    )


class RequestJsonErrorTest(unittest.TestCase):
    def test_structured_error_body_surfaces_its_message(self):
        body = '{"detail": {"error": "budget_exhausted", "message": "Ran out of GPU memory."}}'
        with mock.patch.object(hook.urllib.request, "urlopen", side_effect=make_http_error(503, body)):
            with self.assertRaises(RuntimeError) as ctx:
                hook.request_json("/check", {"text": "hi"})
        message = str(ctx.exception)
        self.assertIn("HTTP 503", message)
        self.assertIn("Ran out of GPU memory.", message)
        self.assertNotIn("not reachable", message)

    def test_http_error_surfaces_status_and_detail(self):
        oom = '{"detail": "CUDA out of memory while scanning."}'
        with mock.patch.object(hook.urllib.request, "urlopen", side_effect=make_http_error(503, oom)):
            with self.assertRaises(RuntimeError) as ctx:
                hook.request_json("/check", {"text": "hi"})
        message = str(ctx.exception)
        self.assertIn("HTTP 503", message)
        self.assertIn("CUDA out of memory", message)
        self.assertNotIn("not reachable", message)

    def test_plain_500_body_is_included(self):
        with mock.patch.object(hook.urllib.request, "urlopen", side_effect=make_http_error(500, "Internal Server Error")):
            with self.assertRaises(RuntimeError) as ctx:
                hook.request_json("/check", {"text": "hi"})
        message = str(ctx.exception)
        self.assertIn("HTTP 500", message)
        self.assertIn("Internal Server Error", message)
        self.assertNotIn("not reachable", message)

    def test_connection_error_still_reports_unreachable(self):
        with mock.patch.object(hook.urllib.request, "urlopen", side_effect=urllib.error.URLError("refused")):
            with self.assertRaises(RuntimeError) as ctx:
                hook.request_json("/health")
        self.assertIn("not reachable", str(ctx.exception))

    def test_timeout_is_not_reported_as_unreachable(self):
        # The service is alive and probably still scanning; "start it" is wrong.
        with mock.patch.object(hook.urllib.request, "urlopen", side_effect=TimeoutError()):
            with self.assertRaises(hook.ScanTimeout) as ctx:
                hook.request_json("/check", {"text": "hi"})
        message = str(ctx.exception)
        self.assertIn("did not answer", message)
        self.assertNotIn("not reachable", message)

    def test_health_uses_the_short_timeout(self):
        seen = {}

        def record(request, timeout=None):
            seen["timeout"] = timeout
            raise urllib.error.URLError("refused")

        with mock.patch.object(hook.urllib.request, "urlopen", side_effect=record):
            with self.assertRaises(hook.ServiceUnreachable):
                hook.request_json("/health")
        self.assertEqual(seen["timeout"], hook.HEALTH_TIMEOUT)

    def test_scan_uses_the_long_timeout(self):
        seen = {}

        def record(request, timeout=None):
            seen["timeout"] = timeout
            raise urllib.error.URLError("refused")

        with mock.patch.object(hook.urllib.request, "urlopen", side_effect=record):
            with self.assertRaises(hook.ServiceUnreachable):
                hook.request_json("/check", {"text": "hi"})
        self.assertEqual(seen["timeout"], hook.REQUEST_TIMEOUT)


class AutostartTest(unittest.TestCase):
    """The hook delegates start-and-wait to service_control.

    The polling, the lock and the "reuse a running service instead of replacing
    it" rule are covered in test_service_control.py; what matters here is that
    the hook asks for it, and that it reports ownership correctly so it only
    stops a service it started itself.
    """

    def test_autostart_takes_a_lease(self):
        # The lease is what stops a concurrent commit from tearing the service
        # down mid-scan, so the hook must always ask for one.
        with mock.patch.object(hook, "AUTOSTART", True), \
                mock.patch.object(
                    hook.service_control, "state", return_value=hook.service_control.STATE_DOWN
                ), \
                mock.patch.object(hook.service_control, "ensure", return_value="started") as ensure:
            self.assertTrue(hook.check_service())
        ensure.assert_called_once_with(hook.STARTUP_TIMEOUT, lease=True)

    def test_reusing_a_running_service_still_takes_a_lease(self):
        # Even when another caller started it, this commit is a user of it and
        # must be counted, or the starter's release would stop it mid-scan.
        with mock.patch.object(hook, "AUTOSTART", True), \
                mock.patch.object(
                    hook.service_control, "state", return_value=hook.service_control.STATE_HEALTHY
                ), \
                mock.patch.object(hook.service_control, "ensure", return_value="reused") as ensure:
            self.assertTrue(hook.check_service())
        ensure.assert_called_once_with(hook.STARTUP_TIMEOUT, lease=True)

    def test_autostart_disabled_propagates_unreachable(self):
        with mock.patch.object(hook, "AUTOSTART", False), \
                mock.patch.object(
                    hook.service_control, "state", return_value=hook.service_control.STATE_DOWN
                ), \
                mock.patch.object(hook.service_control, "ensure") as ensure:
            with self.assertRaises(hook.ServiceUnreachable):
                hook.check_service()
        ensure.assert_not_called()

    def test_autostart_disabled_still_requires_a_gpu(self):
        with mock.patch.object(hook, "AUTOSTART", False), \
                mock.patch.object(
                    hook.service_control, "state", return_value=hook.service_control.STATE_NO_GPU
                ):
            with self.assertRaises(RuntimeError) as ctx:
                hook.check_service()
        self.assertIn("not GPU-backed", str(ctx.exception))

    def test_autostart_disabled_accepts_a_healthy_service(self):
        with mock.patch.object(hook, "AUTOSTART", False), \
                mock.patch.object(
                    hook.service_control, "state", return_value=hook.service_control.STATE_HEALTHY
                ):
            self.assertFalse(hook.check_service())

    def test_failed_start_blocks_with_a_clear_error(self):
        failure = hook.service_control.LifecycleError(
            hook.service_control.EXIT_START_FAILED, "start script failed (exit 1)"
        )
        with mock.patch.object(hook, "AUTOSTART", True), \
                mock.patch.object(
                    hook.service_control, "state", return_value=hook.service_control.STATE_HEALTHY
                ), \
                mock.patch.object(hook.service_control, "ensure", side_effect=failure):
            with self.assertRaises(RuntimeError) as ctx:
                hook.check_service()
        self.assertIn("start script failed", str(ctx.exception))

    def test_service_without_gpu_blocks_the_commit(self):
        failure = hook.service_control.LifecycleError(
            hook.service_control.EXIT_NO_GPU, "no CUDA device"
        )
        with mock.patch.object(hook, "AUTOSTART", True), \
                mock.patch.object(
                    hook.service_control, "state", return_value=hook.service_control.STATE_HEALTHY
                ), \
                mock.patch.object(hook.service_control, "ensure", side_effect=failure):
            with self.assertRaises(RuntimeError) as ctx:
                hook.check_service()
        self.assertIn("no CUDA device", str(ctx.exception))


class AutostopTest(unittest.TestCase):
    def test_stops_service_it_started(self):
        with mock.patch.object(hook, "AUTOSTOP", True), \
                mock.patch.object(hook, "check_service", return_value=True), \
                mock.patch.object(hook, "scan_staged", return_value=0), \
                mock.patch.object(hook, "stop_service") as stop:
            self.assertEqual(hook.main(), 0)
        stop.assert_called_once()

    def test_leaves_user_started_service_running(self):
        with mock.patch.object(hook, "AUTOSTOP", True), \
                mock.patch.object(hook, "check_service", return_value=False), \
                mock.patch.object(hook, "scan_staged", return_value=0), \
                mock.patch.object(hook, "stop_service") as stop:
            self.assertEqual(hook.main(), 0)
        stop.assert_not_called()

    def test_autostop_disabled_keeps_service_running(self):
        with mock.patch.object(hook, "AUTOSTOP", False), \
                mock.patch.object(hook, "check_service", return_value=True), \
                mock.patch.object(hook, "scan_staged", return_value=0), \
                mock.patch.object(hook, "stop_service") as stop:
            self.assertEqual(hook.main(), 0)
        stop.assert_not_called()

    def test_stops_even_when_scan_errors(self):
        with mock.patch.object(hook, "AUTOSTOP", True), \
                mock.patch.object(hook, "check_service", return_value=True), \
                mock.patch.object(hook, "scan_staged", side_effect=RuntimeError("HTTP 503: CUDA out of memory")), \
                mock.patch.object(hook, "stop_service") as stop:
            self.assertEqual(hook.main(), 1)  # fail closed
        stop.assert_called_once()


class ServiceErrorDetailTest(unittest.TestCase):
    def test_extracts_detail_field(self):
        self.assertEqual(hook.service_error_detail('{"detail": "boom"}'), ": boom")

    def test_extracts_message_from_structured_detail(self):
        body = '{"detail": {"error": "no_gpu", "message": "not GPU-backed"}}'
        self.assertEqual(hook.service_error_detail(body), ": not GPU-backed")

    def test_falls_back_to_error_code_without_message(self):
        self.assertEqual(hook.service_error_detail('{"detail": {"error": "no_gpu"}}'), ": no_gpu")

    def test_falls_back_to_raw_body(self):
        self.assertEqual(hook.service_error_detail("plain text"), ": plain text")

    def test_empty_body_yields_no_detail(self):
        self.assertEqual(hook.service_error_detail(""), "")


if __name__ == "__main__":
    unittest.main()
