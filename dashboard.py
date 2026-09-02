"""hearthagent-pro dashboard -- reliability, routing, and eval results,
built from real session data. Run with: uv run streamlit run dashboard.py
"""
import pandas as pd
import streamlit as st

from agent import metrics

st.set_page_config(page_title="hearthagent-pro dashboard", layout="wide")
st.title("hearthagent-pro -- reliability and usage dashboard")
st.caption("Real data logged from actual sessions. Zero cost, fully local.")

turns = metrics.all_turns()
eval_results = metrics.all_eval_results()

if not turns:
    st.warning("No usage data yet. Run some `muni` tasks first, then refresh this page.")
else:
    df = pd.DataFrame(turns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total turns", len(df))
    col2.metric("Total tokens", f"{int(df['input_tokens'].sum() + df['output_tokens'].sum()):,}")
    col3.metric("Avg response time", f"{df['duration_seconds'].mean():.1f}s")
    hit_rate = (df["memory_pre_hit"].sum() / len(df)) * 100
    col4.metric("Memory hit rate", f"{hit_rate:.0f}%")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Task category")
        st.bar_chart(pd.Series(metrics.category_breakdown()))
    with c2:
        st.subheader("Model usage")
        st.bar_chart(pd.Series(metrics.model_breakdown()))
    with c3:
        st.subheader("Memory tier resolution")
        st.bar_chart(pd.Series(metrics.memory_tier_breakdown()))

    st.subheader("Assertion flags caught")
    flagged = df[df["assertion_flags"] != ""]
    if flagged.empty:
        st.info("No assertion flags triggered in logged sessions.")
    else:
        all_flags = []
        for f in flagged["assertion_flags"]:
            all_flags.extend(f.split(","))
        st.bar_chart(pd.Series(all_flags).value_counts())
        with st.expander("Flagged turns detail"):
            st.dataframe(
                flagged[["task_snippet", "category", "model", "assertion_flags"]],
                use_container_width=True,
            )

    st.subheader("Routing drift (flag only -- never changes routing)")
    st.caption(
        "Newer vs older half of each category's escalation-rate window. A flagged "
        "row means the cheap tier started failing more often with no declared config "
        "change (e.g. a tool-description rewrite the fingerprint can't see). Span is "
        "how far back the comparison reaches -- the window is turn-count sized, not "
        "time-bounded, so a wide span on a low-volume route means a stale baseline."
    )
    drift_rows = []
    for cat in metrics.category_breakdown():
        d = metrics.detect_within_window_drift(cat)
        span_days = d.get("window_span_seconds", 0) / 86400
        if d["reason"] == "insufficient_data":
            status = f"-- (not enough data: {d['newer_half_size']}+{d['older_half_size']})"
            newer = older = mag = None
        else:
            status = "DRIFT" if d["drift_detected"] else "stable"
            newer, older, mag = d["newer_half_rate"], d["older_half_rate"], d["drift_magnitude"]
        drift_rows.append({
            "category": cat, "status": status,
            "newer_half_rate": newer, "older_half_rate": older,
            "drift_magnitude": mag, "window_span_days": round(span_days, 1),
        })
    if drift_rows:
        st.dataframe(pd.DataFrame(drift_rows), use_container_width=True, hide_index=True)

    st.subheader("Recent turns")
    st.dataframe(
        df[["timestamp", "category", "model", "duration_seconds",
            "tool_call_count", "memory_tier", "assertion_flags"]]
        .sort_values("timestamp", ascending=False).head(20),
        use_container_width=True,
    )

st.divider()
st.subheader("Eval results (LLM-judged regression tests)")
if not eval_results:
    st.info("No evals run yet. Type 'save eval' in a session after confirming a good answer, "
            "then run: uv run python3 -m bin.evals")
else:
    edf = pd.DataFrame(eval_results)
    pass_rate = (edf["passed"].sum() / len(edf)) * 100
    st.metric("Eval pass rate", f"{pass_rate:.0f}% ({int(edf['passed'].sum())}/{len(edf)})")
    st.dataframe(edf[["task", "passed", "judge_reasoning"]], use_container_width=True)
