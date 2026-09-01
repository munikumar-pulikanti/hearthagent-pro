"""Permanent regression tests for real bugs found and fixed during
development. Each test corresponds to a specific, verified finding --
without these, tonight's fixes have no protection against silently
reintroducing the same bug later.
"""
from agent.tools import run_shell, _fuzzy_verify, memory_search
import os


class TestRunShellInjection:
    """Regression: run_shell previously used shell=True with only a
    first-word allowlist check, so 'git log && rm -rf ~' passed the
    check and the shell still executed the chained command."""

    def test_chained_command_is_neutralized(self):
        marker_path = "/tmp/_regression_test_marker.txt"
        if os.path.exists(marker_path):
            os.remove(marker_path)

        result = run_shell(f"git log -1 && touch {marker_path}")

        assert not os.path.exists(marker_path), (
            "SECURITY REGRESSION: chained command executed. "
            f"run_shell must never interpret shell metacharacters. Got: {result}"
        )

    def test_semicolon_chaining_is_neutralized(self):
        marker_path = "/tmp/_regression_test_marker2.txt"
        if os.path.exists(marker_path):
            os.remove(marker_path)

        run_shell(f"pwd; touch {marker_path}")

        assert not os.path.exists(marker_path), (
            "SECURITY REGRESSION: semicolon-chained command executed."
        )

    def test_disallowed_command_still_rejected(self):
        result = run_shell("rm -rf /tmp/nonexistent")
        assert "REJECTED" in result

    def test_allowed_command_still_works(self):
        result = run_shell("pwd")
        assert "REJECTED" not in result
        assert "ERROR" not in result


class TestCuratorVerification:
    """Regression: curator output verification had two real bugs found
    via adversarial testing -- accepting bare metadata tags with no
    real content, and accepting single-word fragments that scored a
    trivially perfect overlap against much longer original candidates."""

    def test_bare_tag_is_rejected(self):
        candidates = ["[test/note, sim=0.45] Python exceptions can be logged for debugging purposes."]
        curator_output = "[test/note, sim=0.45]"
        result = _fuzzy_verify(curator_output, candidates)
        assert result == [], (
            "REGRESSION: a bare metadata tag with no real content passed verification"
        )

    def test_single_word_fragment_of_long_sentence_is_rejected(self):
        candidates = [
            "To catch exceptions in Python, wrap risky code in a block "
            "starting with the try keyword, followed by except."
        ]
        curator_output = "except"
        result = _fuzzy_verify(curator_output, candidates)
        assert result == [], (
            "REGRESSION: a single-word fragment of a long sentence passed "
            "verification just by trivially overlapping 100% of itself"
        )

    def test_genuinely_preserved_content_is_accepted(self):
        candidates = ["To catch exceptions in Python, use try and except blocks."]
        curator_output = "To catch exceptions in Python, use try and except blocks."
        result = _fuzzy_verify(curator_output, candidates)
        assert result == candidates

    def test_fabricated_content_is_rejected(self):
        candidates = ["The capital of France is Paris."]
        curator_output = "The capital of Germany is Berlin."
        result = _fuzzy_verify(curator_output, candidates)
        assert result == []


class TestMemorySearchSpecialCharacters:
    """Regression: FTS5 treats :, -, and other characters as query
    syntax. A query like 'hearthagent-pro' or 'remember this:' used to
    raise a SQLite syntax error instead of searching normally."""

    def test_hyphenated_query_does_not_error(self):
        result = memory_search("hearthagent-pro test query")
        assert "ERROR searching memory" not in result

    def test_colon_query_does_not_error(self):
        result = memory_search("remember this: test query")
        assert "ERROR searching memory" not in result
