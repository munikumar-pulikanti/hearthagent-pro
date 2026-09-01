"""Regression tests for cascade reliability checks and metrics fixes
found and fixed during development."""
import sys
import uuid
sys.path.insert(0, ".")
from agent.graph import check_tool_result_fidelity
from agent import metrics


class FakeToolMessage:
    """Minimal stand-in for a LangGraph tool message -- the fidelity
    check only reads .name and .content, no need for the real class."""
    def __init__(self, name, content):
        self.name = name
        self.content = content


class TestToolResultFidelity:
    """Regression: found via real testing that llama3.2:1b correctly
    executed list_dir_tool, received correct real data back, then
    fabricated additional filenames in its own summary of that real
    result. The tool executing correctly says nothing about whether the
    model's narration of the result stayed honest."""

    def test_fabricated_filenames_are_flagged(self):
        real_output = "f .env\nd .git\nf README.md\nf main.py"
        messages = [FakeToolMessage("list_dir_tool", real_output)]
        summary_with_fabrication = (
            "The files are: .env, .git, README.md, main.py, .env.dev, .env.test"
        )
        flags = check_tool_result_fidelity(summary_with_fabrication, messages)
        assert any("fabricated_items_in_summary" in f for f in flags), (
            "REGRESSION: a summary listing files that never appeared in the "
            "real tool output was not flagged as fabricated"
        )

    def test_accurate_summary_is_not_flagged(self):
        real_output = "f .env\nd .git\nf README.md\nf main.py"
        messages = [FakeToolMessage("list_dir_tool", real_output)]
        accurate_summary = "The files are: .env, .git, README.md, main.py"
        flags = check_tool_result_fidelity(accurate_summary, messages)
        assert flags == [], (
            "A summary that only mentions real items from the tool output "
            "should not be flagged"
        )

    def test_untracked_tool_is_ignored(self):
        messages = [FakeToolMessage("some_other_tool", "irrelevant content here")]
        flags = check_tool_result_fidelity("anything at all", messages)
        assert flags == [], "Only FIDELITY_CHECKED_TOOLS should ever trigger this check"


class TestShortcutSelfReinforcement:
    """Regression: category_escalation_rate previously counted
    shortcut-skipped turns (cheap tier never attempted, logged as
    escalated=True by construction) the same as genuine cheap-tier
    failures. This made the rate self-reinforcing: shortcut fires ->
    more fake-escalated rows -> rate stays pinned high forever, even if
    the underlying model improved."""

    def test_shortcut_skips_are_excluded_from_rate_calculation(self):
        test_category = f"_test_regressions_shortcut_{uuid.uuid4().hex}"

        # A real cheap-tier attempt that succeeded (not escalated)
        metrics.log_turn(
            task_snippet="real cheap success", category=test_category,
            model="test-model", duration_seconds=1.0,
            cascade_tier="cheap", escalated=False,
            cheap_attempt_tokens=100, capable_attempt_tokens=0,
        )

        # A shortcut-skip: cheap tier never ran (cheap_attempt_tokens=None),
        # but escalated=True by construction since capable served it
        metrics.log_turn(
            task_snippet="shortcut skip", category=test_category,
            model="test-model", duration_seconds=1.0,
            cascade_tier="capable", escalated=True, shortcut_fired=True,
            cheap_attempt_tokens=None, capable_attempt_tokens=200,
        )

        stats = metrics.category_escalation_rate(test_category)

        assert stats["sample_size"] == 1, (
            f"REGRESSION: expected the shortcut-skip row to be excluded from "
            f"the rate calculation, but sample_size was {stats['sample_size']} "
            f"(should only count the 1 real cheap-tier attempt)"
        )
        assert stats["escalation_rate"] == 0.0, (
            "REGRESSION: the real cheap-tier attempt succeeded (not escalated), "
            "so the rate should be 0.0 once the fake shortcut-escalation is "
            "correctly excluded"
        )
