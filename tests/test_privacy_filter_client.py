"""Tests for the CLI's error taxonomy and exit codes.

A caller that pipes secrets through this tool has to be able to tell "clean" from
"could not scan"; every failure mode therefore gets its own exit code, and none
of them tells anyone to skip the scan. Every fixture here is synthetic.

Run: python3 -m unittest discover -s tests
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import privacy_filter_client as client  # noqa: E402


def http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://127.0.0.1:8757/check",
        code=code,
        msg="error",
        hdrs=None,
        fp=io.BytesIO(body.encode("utf-8")),
    )


def run_cli(argv: list[str], response=None, error=None) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    if error is not None:
        patch = mock.patch.object(client, "request_json", side_effect=error)
    else:
        patch = mock.patch.object(client, "request_json", return_value=response)
    with patch, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = client.main(argv)
    return code, out.getvalue(), err.getvalue()


class ExitCodeTest(unittest.TestCase):
    def write_input(self, text: str) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        handle.write(text)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_clean_check_exits_zero(self):
        path = self.write_input("nothing private here\n")
        code, out, _ = run_cli(["check", "--file", path], response={"count": 0, "findings": []})
        self.assertEqual(code, client.EXIT_CLEAN)
        self.assertIn('"count": 0', out)

    def test_findings_exit_one(self):
        path = self.write_input("Ada was here\n")
        response = {"count": 1, "findings": [{"label": "private_person", "start": 0, "end": 3}]}
        code, _, _ = run_cli(["check", "--file", path], response=response)
        self.assertEqual(code, client.EXIT_FINDINGS)

    def test_missing_file_is_invalid_input(self):
        code, _, err = run_cli(["check", "--file", "/nonexistent/input.txt"], response={})
        self.assertEqual(code, client.EXIT_INVALID_INPUT)
        self.assertIn("No such file", err)

    def test_undecodable_file_is_invalid_input(self):
        handle = tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False)
        handle.write(b"\xff\xfe\x00binary")
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        code, _, err = run_cli(["check", "--file", handle.name], response={})
        self.assertEqual(code, client.EXIT_INVALID_INPUT)
        self.assertIn("not valid UTF-8", err)

    def test_unreachable_service_has_its_own_code(self):
        path = self.write_input("text\n")
        code, _, err = run_cli(
            ["check", "--file", path], error=client.ServiceUnreachable("not reachable")
        )
        self.assertEqual(code, client.EXIT_UNREACHABLE)
        self.assertIn("not reachable", err)

    def test_missing_gpu_has_its_own_code(self):
        path = self.write_input("text\n")
        error = client.ServiceError(503, "no_gpu", "Privacy Filter is not GPU-backed")
        code, _, err = run_cli(["check", "--file", path], error=error)
        self.assertEqual(code, client.EXIT_NO_GPU)
        self.assertIn("not GPU-backed", err)

    def test_exhausted_budget_has_its_own_code(self):
        path = self.write_input("text\n")
        error = client.ServiceError(503, "budget_exhausted", "Ran out of GPU memory")
        code, _, err = run_cli(["check", "--file", path], error=error)
        self.assertEqual(code, client.EXIT_BUDGET_EXHAUSTED)
        self.assertIn("Ran out of GPU memory", err)

    def test_timeout_is_not_reported_as_unreachable(self):
        # A slow scan and a stopped service need opposite responses; conflating
        # them sends people restarting a service that is busy working.
        path = self.write_input("text\n")
        code, _, err = run_cli(
            ["check", "--file", path], error=client.ScanTimeout("did not finish within 900s")
        )
        self.assertEqual(code, client.EXIT_TIMEOUT)
        self.assertIn("did not finish", err)
        self.assertNotIn("not reachable", err)

    def test_unknown_service_error_falls_back(self):
        path = self.write_input("text\n")
        code, _, _ = run_cli(
            ["check", "--file", path], error=client.ServiceError(500, "", "boom")
        )
        self.assertEqual(code, client.EXIT_SERVICE_ERROR)

    def test_redact_prints_redacted_text(self):
        path = self.write_input("Ada was here\n")
        response = {"count": 1, "findings": [], "redacted": "[PRIVATE_PERSON] was here\n"}
        code, out, _ = run_cli(["redact", "--file", path], response=response)
        self.assertEqual(code, client.EXIT_CLEAN)
        self.assertEqual(out, "[PRIVATE_PERSON] was here\n")


class RequestJsonTest(unittest.TestCase):
    def test_structured_error_body_becomes_a_typed_error(self):
        body = '{"detail": {"error": "budget_exhausted", "message": "no memory"}}'
        with mock.patch.object(client.urllib.request, "urlopen", side_effect=http_error(503, body)):
            with self.assertRaises(client.ServiceError) as ctx:
                client.request_json("http://127.0.0.1:8757", "/check", {"text": "hi"})
        self.assertEqual(ctx.exception.code, "budget_exhausted")
        self.assertEqual(ctx.exception.status, 503)
        self.assertIn("no memory", str(ctx.exception))

    def test_plain_error_body_is_kept(self):
        with mock.patch.object(
            client.urllib.request, "urlopen", side_effect=http_error(500, "Internal Server Error")
        ):
            with self.assertRaises(client.ServiceError) as ctx:
                client.request_json("http://127.0.0.1:8757", "/check", {"text": "hi"})
        self.assertEqual(ctx.exception.code, "")
        self.assertIn("Internal Server Error", str(ctx.exception))

    def test_connection_failure_is_unreachable(self):
        with mock.patch.object(
            client.urllib.request, "urlopen", side_effect=urllib.error.URLError("refused")
        ):
            with self.assertRaises(client.ServiceUnreachable):
                client.request_json("http://127.0.0.1:8757", "/health")

    def test_read_timeout_is_a_scan_timeout(self):
        with mock.patch.object(client.urllib.request, "urlopen", side_effect=TimeoutError()):
            with self.assertRaises(client.ScanTimeout) as ctx:
                client.request_json("http://127.0.0.1:8757", "/check", {"text": "hi"}, timeout=5)
        self.assertIn("5s", str(ctx.exception))

    def test_connect_timeout_wrapped_in_urlerror_is_a_scan_timeout(self):
        error = urllib.error.URLError(TimeoutError("timed out"))
        with mock.patch.object(client.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(client.ScanTimeout):
                client.request_json("http://127.0.0.1:8757", "/check", {"text": "hi"})

    def test_parse_error_body_variants(self):
        self.assertEqual(client.parse_error_body(""), ("", ""))
        self.assertEqual(client.parse_error_body("plain"), ("", "plain"))
        self.assertEqual(client.parse_error_body('{"detail": "why"}'), ("", "why"))


class NoCallerChunkingTest(unittest.TestCase):
    def test_whole_input_is_submitted_in_one_request(self):
        # The service owns windowing; the CLI must not pre-chunk or truncate.
        text = "line of text\n" * 100_000
        handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        handle.write(text)
        handle.close()
        self.addCleanup(os.unlink, handle.name)

        calls = []

        def record(base_url, path, payload=None, timeout=None):
            calls.append(payload)
            return {"count": 0, "findings": []}

        with mock.patch.object(client, "request_json", side_effect=record), \
                contextlib.redirect_stdout(io.StringIO()):
            code = client.main(["check", "--file", handle.name])

        self.assertEqual(code, client.EXIT_CLEAN)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["text"], text)


if __name__ == "__main__":
    unittest.main()
