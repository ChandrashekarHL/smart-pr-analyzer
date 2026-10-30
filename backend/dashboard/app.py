"""prana review dashboard (design doc FR-10).

    streamlit run backend/dashboard/app.py

Reads everything from the backend's `/api/dashboard` (bearer token), so the
dashboard holds no database credentials. Behind Caddy at /dashboard with
basic auth in production.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import data  # noqa: E402

st.set_page_config(page_title="prana · review dashboard", page_icon="🔍", layout="wide",
                   initial_sidebar_state="expanded")

SEVERITY_COLORS = alt.Scale(domain=data.SEVERITY_ORDER, range=["#d1242f", "#bf8700", "#0969da"])

api_url, token, wandb_url = data.settings()

with st.sidebar:
    st.header("prana")
    days = st.selectbox("Window", [7, 30, 90], index=1, format_func=lambda d: f"Last {d} days")
    if st.button("Refresh"):
        st.cache_data.clear()
    st.caption("Severity and issue type are hints, not merge gates.")


@st.cache_data(ttl=60, show_spinner=False)
def load(url: str, _token: str, days: int) -> dict:
    # The backend URL is part of the cache key; the token (underscore) is not
    # hashed into it, so it never lands in Streamlit's cache keys.
    return data.fetch(url, _token, days)


st.title("PR review dashboard")
try:
    d = load(api_url, token, days)
except data.DashboardError as e:
    st.error(f"Cannot load dashboard data: {e}")
    st.stop()

st.caption(f"Last {d['window_days']} days · loaded "
           f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
t = d["totals"]
if t["runs"] == 0:
    st.info("No reviews in this window yet. Open a pull request on a repository where the "
            "GitHub App is installed.")
    st.stop()

# ---------------------------------------------------------------- KPIs
fb = d["feedback"]
lat = d["latency"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Reviews posted", t["reviews_posted"], help="Pull request reviews posted on GitHub")
c2.metric("Inline comments", t["comments_inline"],
          help=f"{t['comments_held_back']} more held back in summaries (low confidence)")
c3.metric("p95 review time", f"{lat['p95_s']:.1f} s" if lat["p95_s"] is not None else "–",
          delta=data.p95_delta(d), delta_color="inverse")
c4.metric("Helpful (👍 share)",
          f"{fb['helpful_rate']:.0%}" if fb["helpful_rate"] is not None else "–",
          help=f"{fb['thumbs_up']} 👍 · {fb['thumbs_down']} 👎 on "
               f"{fb['rated_comments']} rated comments")
c5.metric("Failed runs", t["failed"], help="Kept as dead letters; replay with prana_app.admin")

# ---------------------------------------------------------------- charts
left, right = st.columns(2)
with left:
    st.subheader("Review time per day")
    rows = data.latency_rows(d)
    if rows:
        df = pd.DataFrame(rows)
        line = alt.Chart(df).mark_line(point=True).encode(
            # One point per day: a date axis, not hours.
            x=alt.X("yearmonthdate(day):T", title=None,
                    axis=alt.Axis(format="%d %b", tickCount="day")),
            y=alt.Y("seconds:Q", title="seconds"),
            color=alt.Color("measure:N", title=None))
        target = alt.Chart(pd.DataFrame({"y": [data.TARGET_P95_S]})).mark_rule(
            strokeDash=[4, 4], color="#d1242f").encode(y="y:Q")
        st.altair_chart(line + target, width="stretch")
        st.caption("Dashed line: the 30 s per-PR target.")
    else:
        st.write("No timed reviews yet.")
with right:
    st.subheader("Comments by severity")
    sev = pd.DataFrame(data.severity_rows(d))
    st.altair_chart(alt.Chart(sev).mark_bar().encode(
        x=alt.X("severity:N", sort=data.SEVERITY_ORDER, title=None),
        y=alt.Y("comments:Q", title="comments"),
        color=alt.Color("severity:N", scale=SEVERITY_COLORS, legend=None)),
        width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Comments by issue type")
    it = pd.DataFrame([{"issue type": k, "comments": v}
                       for k, v in d["comments_by_issue_type"].items()])
    if len(it):
        st.altair_chart(alt.Chart(it).mark_bar().encode(
            y=alt.Y("issue type:N", sort="-x", title=None), x=alt.X("comments:Q")),
            width="stretch")
with right:
    st.subheader("Model confidence")
    st.altair_chart(alt.Chart(pd.DataFrame(d["confidence_histogram"])).mark_bar().encode(
        x=alt.X("bin:N", title="confidence", sort=None), y=alt.Y("comments:Q")),
        width="stretch")
    st.caption("Confidence is the model's mean token probability, not a calibrated score.")

# ---------------------------------------------------------------- tables
st.subheader("Recent reviews")
runs = pd.DataFrame(d["recent_runs"])
runs["created_at"] = pd.to_datetime(runs["created_at"], utc=True)
st.dataframe(runs, hide_index=True, width="stretch",
             column_config={
                 "seconds": st.column_config.NumberColumn(format="%.1f s"),
                 "created_at": st.column_config.DatetimeColumn("created (UTC)",
                                                               format="YYYY-MM-DD HH:mm"),
                 "hunks": st.column_config.TextColumn(help="reviewed / changed hunks")})

st.subheader("Recent comments")
st.dataframe(pd.DataFrame(d["recent_comments"]), hide_index=True, width="stretch",
             column_config={"confidence": st.column_config.ProgressColumn(
                 min_value=0.0, max_value=1.0, format="%.2f")})

# ---------------------------------------------------------------- W&B
st.subheader("Training and evaluation")
if wandb_url:
    st.components.v1.iframe(wandb_url, height=600, scrolling=True)
else:
    st.caption("Set WANDB_REPORT_URL to embed the public Weights & Biases report here.")
