"""Tests for what the pre-commit hook refuses to leave unscanned.

The hook's whole value is that nothing slips through: an oversized diff, a binary
blob or an undecodable path must block the commit, not print a warning and let it
land. Every fixture here is synthetic.

Run: python3 -m unittest discover -s tests
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import git_pre_commit_pii as hook  # noqa: E402


def payload(path: str, text: str) -> hook.ScanPayload:
    locations = [
        hook.OffsetLocation(path=path, line=1, column=index + 1) for index in range(len(text))
    ]
    return hook.ScanPayload(path=path, text=text, locations=locations)


class StagedScanTest(unittest.TestCase):
    def stage(self, payloads, binaries=(), allowlist=(), reviewed=(), request=None):
        """Run scan_staged over synthetic staged files, returning (code, requests)."""
        self.printed = ""
        requests: list[dict] = []

        def fake_request(path, body=None):
            requests.append({"path": path, "body": body})
            if request is not None:
                return request(path, body)
            return {"count": 0, "findings": []}

        by_path = {item.path: item for item in payloads}
        paths = [item.path for item in payloads] + [name for name in binaries]
        stderr = io.StringIO()
        with mock.patch.object(hook, "staged_paths", return_value=paths), \
                mock.patch.object(hook, "binary_paths", return_value=set(binaries)), \
                mock.patch.object(hook, "binary_allowlist", return_value=list(allowlist)), \
                mock.patch.object(hook, "reviewed_findings", return_value=set(reviewed)), \
                mock.patch.object(hook, "build_payload", side_effect=lambda path: by_path[path]), \
                mock.patch.object(hook, "request_json", side_effect=fake_request), \
                contextlib.redirect_stderr(stderr):
            code = hook.scan_staged()
        self.printed = stderr.getvalue()
        return code, requests

    def test_oversized_added_text_is_scanned_not_skipped(self):
        # Previously anything over 1 MiB was skipped with a warning; the service
        # now windows it, so the hook submits the whole payload.
        text = ("safe line of ordinary text\n" * 60_000)
        self.assertGreater(len(text.encode("utf-8")), 1024 * 1024)
        code, requests = self.stage([payload("huge.txt", text)])
        self.assertEqual(code, 0)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["body"]["text"], text)

    def test_binary_file_blocks_the_commit(self):
        code, requests = self.stage([], binaries=["logo.png"])
        self.assertEqual(code, 1)
        self.assertEqual(requests, [])

    def test_allowlisted_binary_is_skipped_deliberately(self):
        code, _ = self.stage([], binaries=["assets/logo.png"], allowlist=["*.png"])
        self.assertEqual(code, 0)

    def test_allowlist_does_not_cover_other_binaries(self):
        code, _ = self.stage([], binaries=["secrets.pdf"], allowlist=["*.png"])
        self.assertEqual(code, 1)

    def test_undecodable_diff_blocks_the_commit(self):
        stderr = io.StringIO()
        with mock.patch.object(hook, "staged_paths", return_value=["weird.txt"]), \
                mock.patch.object(hook, "binary_paths", return_value=set()), \
                mock.patch.object(hook, "binary_allowlist", return_value=[]), \
                mock.patch.object(
                    hook, "build_payload", side_effect=hook.UndecodableDiff("not valid UTF-8")
                ), \
                contextlib.redirect_stderr(stderr):
            self.assertEqual(hook.scan_staged(), 1)
        self.assertIn("weird.txt", stderr.getvalue())

    def test_service_failure_blocks_and_stops_scanning(self):
        def failing(path, body=None):
            raise RuntimeError("HTTP 503: out of memory at the minimum token budget")

        code, requests = self.stage(
            [payload("a.txt", "hello there"), payload("b.txt", "hello again")],
            request=failing,
        )
        self.assertEqual(code, 1)
        self.assertEqual(len(requests), 1, "kept scanning after a service failure")

    def test_findings_block_and_report_location_without_text(self):
        def found(path, body=None):
            return {
                "count": 1,
                "findings": [{"label": "private_person", "start": 0, "end": 5, "score": 0.99}],
            }

        code, _ = self.stage([payload("notes.md", "Alice was here")], request=found)
        self.assertEqual(code, 1)
        self.assertIn("notes.md:1:1: private_person", self.printed)
        self.assertNotIn("Alice", self.printed)

    def found_person(self, path, body=None):
        return {
            "count": 1,
            "findings": [{"label": "private_person", "start": 0, "end": 5, "score": 0.99}],
        }

    def test_exact_reviewed_finding_passes_without_disabling_scan(self):
        text = "Alice was here"
        reviewed = hook.ReviewedFinding(
            "package.json", "private_person", hook.finding_digest(text, 0, 5)
        )
        code, requests = self.stage(
            [payload("package.json", text)], reviewed={reviewed}, request=self.found_person
        )

        self.assertEqual(code, 0)
        self.assertEqual(len(requests), 1)
        self.assertIn("reviewed false positive package.json:1:private_person", self.printed)

    def test_different_text_on_the_reviewed_line_still_blocks(self):
        # The fail-open this keying exists to close: an entry reviewed for one
        # value must not suppress a different value that lands in its place.
        reviewed = hook.ReviewedFinding(
            "package.json", "private_person", hook.finding_digest("Alice was here", 0, 5)
        )
        code, _ = self.stage(
            [payload("package.json", "Bruno was here")],
            reviewed={reviewed},
            request=self.found_person,
        )

        self.assertEqual(code, 1, "a different value reused the reviewed entry")
        self.assertIn("package.json:1:1: private_person", self.printed)

    def test_reviewed_entry_does_not_carry_across_labels(self):
        text = "Alice was here"
        reviewed = hook.ReviewedFinding(
            "package.json", "secret", hook.finding_digest(text, 0, 5)
        )
        code, _ = self.stage(
            [payload("package.json", text)], reviewed={reviewed}, request=self.found_person
        )
        self.assertEqual(code, 1)

    def test_reviewed_entry_does_not_carry_across_files(self):
        text = "Alice was here"
        reviewed = hook.ReviewedFinding(
            "other.json", "private_person", hook.finding_digest(text, 0, 5)
        )
        code, _ = self.stage(
            [payload("package.json", text)], reviewed={reviewed}, request=self.found_person
        )
        self.assertEqual(code, 1)

    def test_blocking_report_offers_a_paste_ready_allowlist_line(self):
        # The digest cannot be produced by hand, so the report has to supply it.
        text = "Alice was here"
        code, _ = self.stage([payload("package.json", text)], request=self.found_person)
        self.assertEqual(code, 1)
        expected = f"package.json\tprivate_person\t{hook.finding_digest(text, 0, 5)}"
        self.assertIn(expected, self.printed)
        self.assertNotIn("Alice", self.printed)

    def test_control_files_are_not_scanned(self):
        # Their content is digests and fnmatch patterns; the model reads a hex
        # digest as a secret, so scanning them would demand allowlist entries
        # made of digests that flag in turn.
        code, requests = self.stage(
            [
                payload(hook.REVIEWED_FINDINGS_FILE, "src.ts\tsecret\tabc0123456789def"),
                payload(hook.BINARY_ALLOWLIST_FILE, "*.png"),
            ]
        )
        self.assertEqual(code, 0)
        self.assertEqual(requests, [], "a control file was sent to the model")
        self.assertIn("does not scan its own control file", self.printed)

    def test_skipping_control_files_is_announced(self):
        code, _ = self.stage([payload(hook.REVIEWED_FINDINGS_FILE, "x\ty\tabc0123456789def")])
        self.assertEqual(code, 0)
        self.assertIn(hook.REVIEWED_FINDINGS_FILE, self.printed)

    def test_clean_staged_files_pass(self):
        code, requests = self.stage([payload("clean.txt", "nothing to see")])
        self.assertEqual(code, 0)
        self.assertEqual(len(requests), 1)

    def test_whitespace_only_addition_needs_no_request(self):
        code, requests = self.stage([payload("blank.txt", "\n\n   \n")])
        self.assertEqual(code, 0)
        self.assertEqual(requests, [])


class ReviewedFindingsFileTest(unittest.TestCase):
    def write(self, body: str):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        with open(os.path.join(root, hook.REVIEWED_FINDINGS_FILE), "w", encoding="utf-8") as fh:
            fh.write(body)
        return root

    def test_reads_entries_and_skips_comments(self):
        digest = "0123456789abcdef"
        root = self.write(f"# reviewed 2026-09-16\n\nsrc/app.ts\tprivate_url\t{digest}\n")
        with mock.patch.object(hook, "repo_root", return_value=root):
            self.assertEqual(
                hook.reviewed_findings(),
                {hook.ReviewedFinding("src/app.ts", "private_url", digest)},
            )

    def test_old_line_based_format_is_rejected(self):
        # Honouring it would keep the line-pinned suppression alive.
        root = self.write("test/app.test.ts\t33\tprivate_url\n")
        with mock.patch.object(hook, "repo_root", return_value=root):
            with self.assertRaises(RuntimeError) as ctx:
                hook.reviewed_findings()
        self.assertIn("old path/line/label format", str(ctx.exception))

    def test_malformed_entry_fails_closed(self):
        root = self.write("only-two\tfields\n")
        with mock.patch.object(hook, "repo_root", return_value=root):
            with self.assertRaises(RuntimeError):
                hook.reviewed_findings()

    def test_bad_digest_fails_closed(self):
        root = self.write("src/app.ts\tprivate_url\tnot-a-digest\n")
        with mock.patch.object(hook, "repo_root", return_value=root):
            with self.assertRaises(RuntimeError):
                hook.reviewed_findings()

    def test_missing_file_is_an_empty_set(self):
        with mock.patch.object(hook, "repo_root", return_value="/nonexistent-repo"):
            self.assertEqual(hook.reviewed_findings(), set())


class FindingDigestTest(unittest.TestCase):
    def test_digest_covers_only_the_matched_span(self):
        text = "hello Alice goodbye"
        self.assertEqual(
            hook.finding_digest(text, 6, 11), hook.finding_digest("Alice", 0, 5)
        )

    def test_different_text_gives_a_different_digest(self):
        self.assertNotEqual(
            hook.finding_digest("Alice", 0, 5), hook.finding_digest("Bruno", 0, 5)
        )

    def test_digest_is_short_hex(self):
        digest = hook.finding_digest("Alice", 0, 5)
        self.assertTrue(hook.is_digest(digest))
        self.assertEqual(len(digest), hook.DIGEST_LENGTH)

    def test_is_digest_rejects_junk(self):
        self.assertFalse(hook.is_digest("XYZ"))
        self.assertFalse(hook.is_digest("0123456789abcdeg"))
        self.assertFalse(hook.is_digest("0123456789abcdef0"))


class BinaryAllowlistTest(unittest.TestCase):
    def test_env_patterns_are_split_on_colons_and_commas(self):
        with mock.patch.dict(os.environ, {hook.BINARY_ALLOWLIST_ENV: "*.png, docs/*.pdf"}), \
                mock.patch.object(hook, "repo_root", return_value=""):
            self.assertEqual(hook.binary_allowlist(), ["*.png", "docs/*.pdf"])

    def test_repo_file_patterns_are_read(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, hook.BINARY_ALLOWLIST_FILE), "w", encoding="utf-8") as fh:
                fh.write("# fonts carry no PII\n\n*.woff2\n")
            with mock.patch.dict(os.environ, {hook.BINARY_ALLOWLIST_ENV: ""}), \
                    mock.patch.object(hook, "repo_root", return_value=root):
                self.assertEqual(hook.binary_allowlist(), ["*.woff2"])

    def test_missing_repo_file_is_an_empty_allowlist(self):
        with mock.patch.dict(os.environ, {hook.BINARY_ALLOWLIST_ENV: ""}), \
                mock.patch.object(hook, "repo_root", return_value="/nonexistent-repo-root"):
            self.assertEqual(hook.binary_allowlist(), [])

    def test_patterns_match_full_path_or_basename(self):
        self.assertTrue(hook.is_allowlisted("assets/img/logo.png", ["*.png"]))
        self.assertTrue(hook.is_allowlisted("assets/img/logo.png", ["assets/*/logo.png"]))
        self.assertFalse(hook.is_allowlisted("assets/img/logo.png", ["*.jpg"]))
        self.assertFalse(hook.is_allowlisted("assets/img/logo.png", []))


class DecodeTest(unittest.TestCase):
    def test_non_utf8_path_is_rejected(self):
        with self.assertRaises(hook.UndecodableDiff):
            hook.decode_path(b"caf\xe9.txt")

    def test_utf8_path_decodes(self):
        self.assertEqual(hook.decode_path("café.txt".encode("utf-8")), "café.txt")


class NoBypassAdviceTest(unittest.TestCase):
    """Privacy enforcement is fail-closed; nothing may advertise the escape hatch."""

    SOURCES = ["scripts", "skill", "hooks"]

    def test_no_source_recommends_skipping_verification(self):
        root = os.path.join(os.path.dirname(__file__), "..")
        offenders = []
        for directory in self.SOURCES:
            for dirpath, _, filenames in os.walk(os.path.join(root, directory)):
                if "__pycache__" in dirpath:
                    continue
                for filename in filenames:
                    path = os.path.join(dirpath, filename)
                    with open(path, encoding="utf-8", errors="replace") as handle:
                        if "--no-verify" in handle.read():
                            offenders.append(os.path.relpath(path, root))
        self.assertEqual(offenders, [], "these files suggest bypassing the privacy hook")


if __name__ == "__main__":
    unittest.main()
