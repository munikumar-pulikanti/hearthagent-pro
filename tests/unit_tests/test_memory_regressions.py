"""Regression tests for memory save/embed/confidence behavior found and
fixed during development.

Note on test isolation: corroboration checks similarity GLOBALLY across
all memories, regardless of scope, and semantic embeddings capture
meaning, not exact string identity -- a random UUID suffix on otherwise
identical phrasing still scores 90%+ similar to a prior test run's
content, correctly triggering corroboration (that's the feature working
as designed). Uniqueness alone does not isolate these tests; real
cleanup of prior test data does."""
from agent.tools import memory_save, _get_chroma_collection
import sqlite3
import uuid
from pathlib import Path

VAULT_DB = Path.home() / ".ai-memory-vault" / "global_brain.db"
TEST_SCOPE = "_test_regressions"


def _cleanup_test_scope():
    """Deletes all memories in TEST_SCOPE from both SQLite and ChromaDB,
    guaranteeing a genuinely clean slate regardless of prior test runs'
    residual data -- corroboration's global similarity check means
    unique content alone doesn't provide real isolation."""
    conn = sqlite3.connect(VAULT_DB)
    rows = conn.execute("SELECT id FROM memories WHERE scope = ?", (TEST_SCOPE,)).fetchall()
    ids = [str(r[0]) for r in rows]
    conn.execute("DELETE FROM memories WHERE scope = ?", (TEST_SCOPE,))
    conn.commit()
    conn.close()

    if ids:
        try:
            _get_chroma_collection().delete(ids=ids)
        except Exception:
            pass  # best-effort -- a stale chroma entry for a deleted row is harmless


class TestAutoEmbedOnSave:
    """Regression: memory_save previously only wrote to SQLite. New
    memories were invisible to semantic search and to corroboration
    checks until a manual sync ran -- found independently in both
    hearthagent (baseline) and hearthagent-pro."""

    def setup_method(self):
        _cleanup_test_scope()

    def test_new_memory_is_immediately_embedded(self):
        unique_marker = f"regression_test_autoembed_marker_{uuid.uuid4().hex}"
        result = memory_save(
            scope=TEST_SCOPE, type_="fact",
            content=f"Autoembed regression test content {unique_marker}",
        )
        assert "Saved memory" in result, (
            f"Expected a fresh save (no similar prior memory after cleanup), got: {result}"
        )

        conn = sqlite3.connect(VAULT_DB)
        row = conn.execute(
            "SELECT id FROM memories WHERE content LIKE ? ORDER BY id DESC LIMIT 1",
            (f"%{unique_marker}%",)
        ).fetchone()
        conn.close()
        assert row is not None, "memory was not saved to SQLite at all"
        new_id = row[0]

        collection = _get_chroma_collection()
        chroma_result = collection.get(ids=[str(new_id)])
        assert chroma_result["ids"], (
            "REGRESSION: new memory was saved to SQLite but never embedded "
            "into ChromaDB -- would be invisible to semantic search and "
            "corroboration checks until a manual sync"
        )


class TestEvidenceGatedConfidence:
    """Regression: confidence should never reach 'confirmed' without a
    real, verified evidence_url, no matter how many times a claim is
    restated. Corroboration without evidence should cap at 'suspected'."""

    def setup_method(self):
        _cleanup_test_scope()

    def test_unverified_claim_caps_at_suspected_even_with_corroboration(self):
        marker = f"evidence gating regression test claim alpha beta gamma {uuid.uuid4().hex}"
        result1 = memory_save(scope=TEST_SCOPE, type_="fact", content=marker)
        assert "hypothesis" in result1, (
            f"Expected a fresh hypothesis-tier save (no similar prior memory after cleanup), got: {result1}"
        )

        memory_save(scope=TEST_SCOPE, type_="fact", content=marker)
        result3 = memory_save(scope=TEST_SCOPE, type_="fact", content=marker)

        assert "confidence=confirmed" not in result3, (
            "REGRESSION: a claim with no verified evidence reached 'confirmed' "
            "confidence purely through repeated restating -- confidence must "
            "be gated on real evidence, not corroboration count alone"
        )

    def test_real_verified_evidence_allows_promotion_to_confirmed(self):
        marker = f"evidence gating regression test claim WITH evidence {uuid.uuid4().hex}"
        evidence = "https://docs.python.org/3/"

        memory_save(scope=TEST_SCOPE, type_="fact", content=marker, evidence_url=evidence)
        memory_save(scope=TEST_SCOPE, type_="fact", content=marker, evidence_url=evidence)
        result3 = memory_save(scope=TEST_SCOPE, type_="fact", content=marker, evidence_url=evidence)

        assert "confirmed" in result3, (
            "A claim with real, verified evidence and sufficient corroboration "
            "should be able to reach confirmed confidence"
        )
