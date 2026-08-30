"""
PipelineIQ — Monitoring Dashboard
----------------------------------
Tracks user feedback and usage patterns for the Data Pipeline Copilot.

To run as part of a Streamlit multipage app, drop this file into a
`pages/` folder next to your main `streamlit_app.py`
(e.g. pages/1_📊_Monitoring_Dashboard.py) so it shows up automatically
in the sidebar page switcher.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.feedback import load_feedback

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

APP_NAME = "PipelineIQ"

st.set_page_config(
    page_title=f"{APP_NAME} · Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# BRAND COLORS (matches the main app's gradient theme)
# --------------------------------------------------

COLOR_POSITIVE = "#34d399"   # emerald
COLOR_NEGATIVE = "#fb7185"   # rose
COLOR_ACCENT_1 = "#818cf8"   # indigo
COLOR_ACCENT_2 = "#38bdf8"   # sky
COLOR_MUTED = "#9ca3af"

PLOTLY_TEMPLATE = "plotly_dark"

# --------------------------------------------------
# STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
        .main { padding-top: 1.5rem; }

        .app-tagline {
            font-size: 0.95rem;
            font-weight: 600;
            color: #a5b4fc;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.3rem;
        }
        .app-title {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #818cf8 0%, #38bdf8 60%, #34d399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            display: inline-block;
            margin-bottom: 0.15rem;
        }
        .app-subtitle {
            color: #9ca3af;
            font-size: 1rem;
            margin-bottom: 1rem;
        }

        div[data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 0.9rem 1rem 0.6rem 1rem;
        }

        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown('<div class="app-tagline">📊 Analytics</div>', unsafe_allow_html=True)
st.markdown(f'<div class="app-title">{APP_NAME} Monitoring Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Track user feedback, satisfaction, RAG pipeline '
    "performance, and retrieval activity for the Data Pipeline Copilot.</div>",
    unsafe_allow_html=True,
)

st.divider()

# --------------------------------------------------
# LOAD FEEDBACK
# --------------------------------------------------

feedback_records = load_feedback()

if not feedback_records:
    st.info(
        "📭 No feedback data is available yet. "
        "Use the Copilot and submit 👍 or 👎 feedback first."
    )
    st.stop()

df = pd.DataFrame(feedback_records)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["date"] = df["timestamp"].dt.date

# Ensure metric columns exist and are numeric, so older feedback records
# (recorded before these fields were tracked) don't break any of the
# charts or aggregations below.
metric_columns = [
    "response_time",
    "retrieved_document_count",
    "query_rewriting_time",
    "retrieval_time",
    "reranking_time",
    "llm_generation_time",
]

for column in metric_columns:
    if column not in df.columns:
        df[column] = None
    df[column] = pd.to_numeric(df[column], errors="coerce")

# "technology" is categorical (e.g. Spark, Airflow, Kafka, dbt) rather than
# numeric, so it just needs to exist for older records.
if "technology" not in df.columns:
    df["technology"] = None

# Other non-numeric columns used for the negative-feedback investigation
# section below — also just need to exist for older records.
for column in ["answer", "retrieved_document_titles", "retrieved_document_ids"]:
    if column not in df.columns:
        df[column] = None

# Rows with an unparseable timestamp can't be date-filtered or charted
# over time; drop them rather than letting NaT break comparisons below.
df = df.dropna(subset=["timestamp"])

if df.empty:
    st.warning("No feedback records have a valid timestamp.")
    st.stop()

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

with st.sidebar:
    st.markdown("### 🔍 Filters")

    min_date, max_date = df["date"].min(), df["date"].max()
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    feedback_types = st.multiselect(
        "Feedback type",
        options=sorted(df["feedback"].unique()),
        default=sorted(df["feedback"].unique()),
    )

    known_technologies = sorted(
        t for t in df["technology"].dropna().unique() if t and t != "Unknown"
    )
    if known_technologies:
        selected_technologies = st.multiselect(
            "Technology",
            options=known_technologies,
            default=known_technologies,
        )
    else:
        selected_technologies = None

    st.caption(f"Showing data from **{min_date}** to **{max_date}**")

# Apply filters
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

mask = (
    (df["date"] >= start_date)
    & (df["date"] <= end_date)
    & (df["feedback"].isin(feedback_types))
)
if selected_technologies is not None:
    mask &= df["technology"].isin(selected_technologies) | df["technology"].isna()

df = df[mask]

if df.empty:
    st.warning("No feedback matches the selected filters.")
    st.stop()

# --------------------------------------------------
# SUMMARY METRICS
# --------------------------------------------------

total_feedback = len(df)
positive_feedback = int((df["feedback"] == "positive").sum())
negative_feedback = int((df["feedback"] == "negative").sum())
positive_rate = (positive_feedback / total_feedback * 100) if total_feedback else 0.0

st.subheader("📌 Feedback Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Feedback", total_feedback)
col2.metric("👍 Positive", positive_feedback)
col3.metric("👎 Negative", negative_feedback)
col4.metric("Positive Rate", f"{positive_rate:.1f}%")

st.divider()

# --------------------------------------------------
# RAG PERFORMANCE
# --------------------------------------------------

st.subheader("⚡ RAG Performance")

average_response_time = df["response_time"].mean()
average_documents = df["retrieved_document_count"].mean()

perf_col1, perf_col2 = st.columns(2)

perf_col1.metric(
    "⏱️ Avg Response Time",
    f"{average_response_time:.2f} sec" if pd.notna(average_response_time) else "No data",
)

perf_col2.metric(
    "📚 Avg Retrieved Documents",
    f"{average_documents:.1f}" if pd.notna(average_documents) else "No data",
)

st.divider()

# --------------------------------------------------
# ANSWER QUALITY BY TECHNOLOGY
# --------------------------------------------------

st.subheader("🧠 Answer Quality by Technology")

technology_df = df[df["technology"].notna() & (df["technology"] != "Unknown")].copy()

if technology_df.empty:
    st.info(
        "No technology feedback data is available yet. "
        "New feedback records will automatically include technology detection."
    )
    technology_summary = pd.DataFrame()
else:
    technology_summary = (
        technology_df.groupby("technology")
        .agg(
            Total_Feedback=("feedback", "count"),
            Positive=("feedback", lambda x: (x == "positive").sum()),
            Negative=("feedback", lambda x: (x == "negative").sum()),
        )
        .reset_index()
    )
    technology_summary["Positive Rate (%)"] = (
        technology_summary["Positive"] / technology_summary["Total_Feedback"] * 100
    ).round(1)
    technology_summary = technology_summary.sort_values("Positive Rate (%)", ascending=False)

    st.dataframe(
        technology_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "technology": st.column_config.TextColumn("Technology"),
            "Total_Feedback": st.column_config.NumberColumn("Total Feedback"),
            "Positive Rate (%)": st.column_config.NumberColumn("Positive Rate", format="%.1f%%"),
        },
    )

    fig_tech = go.Figure(
        data=[
            go.Bar(
                x=technology_summary["technology"],
                y=technology_summary["Positive Rate (%)"],
                marker=dict(
                    color=technology_summary["Positive Rate (%)"],
                    colorscale=[[0, COLOR_NEGATIVE], [0.5, "#facc15"], [1, COLOR_POSITIVE]],
                    cmin=0,
                    cmax=100,
                ),
                text=technology_summary["Positive Rate (%)"].round(1),
                texttemplate="%{text}%",
                textposition="outside",
            )
        ]
    )
    fig_tech.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(t=20, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        xaxis_title=None,
        yaxis_title="Positive rate (%)",
        yaxis_range=[0, 105],
    )
    st.plotly_chart(fig_tech, use_container_width=True)

st.divider()

# --------------------------------------------------
# TECHNOLOGY NEEDING ATTENTION
# --------------------------------------------------

st.subheader("⚠️ Technology Needing Attention")

if technology_summary.empty:
    st.caption("Not enough technology-tagged feedback yet to flag anything.")
else:
    attention_technology = technology_summary.sort_values("Positive Rate (%)", ascending=True).iloc[0]
    attention_name = attention_technology["technology"]
    attention_rate = attention_technology["Positive Rate (%)"]

    if attention_rate < 70:
        st.error(
            f"⚠️ **{attention_name}** currently has the lowest positive feedback "
            f"rate: {attention_rate:.1f}%."
        )
    elif attention_rate < 85:
        st.warning(f"⚠️ **{attention_name}** has the lowest quality score at {attention_rate:.1f}%.")
    else:
        st.success("✅ All detected technologies currently have strong positive feedback.")

st.divider()

# --------------------------------------------------
# PIPELINE STAGE LATENCY
# --------------------------------------------------

st.subheader("⏱️ Average Pipeline Stage Latency")

stage_latency = pd.DataFrame(
    {
        "Stage": ["Query Rewriting", "Retrieval", "Reranking", "LLM Generation"],
        "Average Seconds": [
            df["query_rewriting_time"].mean(),
            df["retrieval_time"].mean(),
            df["reranking_time"].mean(),
            df["llm_generation_time"].mean(),
        ],
    }
).dropna(subset=["Average Seconds"])

if stage_latency.empty:
    st.info(
        "No pipeline stage latency data available yet. "
        "This breaks down response_time into query rewriting, retrieval, "
        "reranking, and LLM generation \u2014 it appears once your RAG pipeline "
        "reports per-stage timings to save_feedback()."
    )
else:
    fig_stages = go.Figure(
        data=[
            go.Bar(
                x=stage_latency["Stage"],
                y=stage_latency["Average Seconds"],
                marker=dict(
                    color=[COLOR_ACCENT_1, COLOR_ACCENT_2, "#a78bfa", COLOR_POSITIVE][: len(stage_latency)]
                ),
                text=stage_latency["Average Seconds"].round(2),
                texttemplate="%{text}s",
                textposition="outside",
            )
        ]
    )
    fig_stages.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(t=20, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        xaxis_title=None,
        yaxis_title="Seconds",
    )
    st.plotly_chart(fig_stages, use_container_width=True)

    stage_latency = stage_latency.copy()
    measured_stage_total = stage_latency["Average Seconds"].sum()
    if measured_stage_total > 0:
        stage_latency["Percentage"] = stage_latency["Average Seconds"] / measured_stage_total * 100

    with st.expander("🔍 Pipeline performance breakdown (table)"):
        breakdown = stage_latency.copy()
        breakdown["Average Seconds"] = breakdown["Average Seconds"].round(3)
        if "Percentage" in breakdown.columns:
            breakdown["Percentage"] = breakdown["Percentage"].round(1)
        st.dataframe(
            breakdown,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Percentage": st.column_config.NumberColumn("Share of Pipeline", format="%.1f%%"),
            },
        )

    # ------------------------------------------
    # AUTOMATIC BOTTLENECK DETECTION
    # ------------------------------------------

    st.subheader("🐢 Automatic Bottleneck Detection")

    bottleneck = stage_latency.sort_values("Average Seconds", ascending=False).iloc[0]
    bottleneck_stage = bottleneck["Stage"]
    bottleneck_time = bottleneck["Average Seconds"]
    bottleneck_percentage = bottleneck.get("Percentage", 0.0)

    bcol1, bcol2, bcol3 = st.columns(3)
    bcol1.metric("🐢 Bottleneck", bottleneck_stage)
    bcol2.metric("Average Time", f"{bottleneck_time:.2f} sec")
    bcol3.metric("Pipeline Share", f"{bottleneck_percentage:.1f}%")

    if bottleneck_percentage >= 60:
        st.error(
            f"⚠️ **{bottleneck_stage}** is the major pipeline bottleneck, using "
            f"{bottleneck_percentage:.1f}% of measured pipeline time."
        )
    elif bottleneck_percentage >= 40:
        st.warning(
            f"⚠️ **{bottleneck_stage}** is currently the largest contributor to "
            f"latency at {bottleneck_percentage:.1f}%."
        )
    else:
        st.success("✅ Pipeline latency is relatively balanced across stages.")

st.divider()

# --------------------------------------------------
# FEEDBACK DISTRIBUTION + OVER TIME (side by side)
# --------------------------------------------------

col_left, col_right = st.columns([1, 1.4])

with col_left:
    st.subheader("🥧 Feedback Distribution")

    feedback_counts = df["feedback"].value_counts()
    colors = [
        COLOR_POSITIVE if label == "positive" else COLOR_NEGATIVE
        for label in feedback_counts.index
    ]

    fig_pie = go.Figure(
        data=[
            go.Pie(
                labels=feedback_counts.index,
                values=feedback_counts.values,
                hole=0.55,
                marker=dict(colors=colors),
                textinfo="label+percent",
            )
        ]
    )
    fig_pie.update_layout(
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("📈 Feedback Over Time")

    daily_feedback = (
        df.groupby(["date", "feedback"]).size().reset_index(name="count")
    )

    fig_line = go.Figure()
    for feedback_type, color in (("positive", COLOR_POSITIVE), ("negative", COLOR_NEGATIVE)):
        subset = daily_feedback[daily_feedback["feedback"] == feedback_type]
        if not subset.empty:
            fig_line.add_trace(
                go.Scatter(
                    x=subset["date"],
                    y=subset["count"],
                    mode="lines+markers",
                    name=feedback_type.capitalize(),
                    line=dict(color=color, width=2.5),
                    marker=dict(size=6),
                )
            )

    fig_line.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None,
        yaxis_title="Feedback count",
    )
    st.plotly_chart(fig_line, use_container_width=True)

st.divider()

# --------------------------------------------------
# NEGATIVE FEEDBACK INVESTIGATION
# --------------------------------------------------

st.subheader("🔍 Negative Feedback Investigation")

negative_df = df[df["feedback"] == "negative"].copy()

if negative_df.empty:
    st.success("🎉 No negative feedback has been recorded yet.")
else:
    st.warning(f"Found {len(negative_df)} negative feedback record(s) that may require investigation.")

    investigation_columns = [
        "timestamp",
        "question",
        "rewritten_question",
        "technology",
        "answer",
        "retrieved_document_titles",
        "retrieved_document_ids",
        "response_time",
    ]
    for column in investigation_columns:
        if column not in negative_df.columns:
            negative_df[column] = None

    investigation_df = negative_df[investigation_columns].sort_values("timestamp", ascending=False)

    st.dataframe(
        investigation_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Time", format="D MMM YYYY, HH:mm"),
            "question": st.column_config.TextColumn("Question", width="medium"),
            "rewritten_question": st.column_config.TextColumn("Rewritten Query", width="medium"),
            "technology": st.column_config.TextColumn("Technology"),
            "answer": st.column_config.TextColumn("Answer", width="large"),
            "retrieved_document_titles": st.column_config.ListColumn("Retrieved Docs"),
            "response_time": st.column_config.NumberColumn("Response Time (s)", format="%.2f s"),
        },
    )

st.divider()

# --------------------------------------------------
# DOCUMENTS ASSOCIATED WITH NEGATIVE FEEDBACK
# --------------------------------------------------

st.subheader("📚 Documents Associated With Negative Feedback")

if negative_df.empty:
    st.info("No negative feedback available.")
else:
    document_counts = {}
    for titles in negative_df["retrieved_document_titles"]:
        if not isinstance(titles, list):
            continue
        for title in titles:
            document_counts[title] = document_counts.get(title, 0) + 1

    if not document_counts:
        st.info("Negative feedback records do not yet contain retrieved document information.")
    else:
        document_df = (
            pd.DataFrame(list(document_counts.items()), columns=["Document", "Negative Feedback Count"])
            .sort_values("Negative Feedback Count", ascending=False)
        )

        fig_docs = go.Figure(
            data=[
                go.Bar(
                    x=document_df["Negative Feedback Count"],
                    y=document_df["Document"],
                    orientation="h",
                    marker=dict(color=COLOR_NEGATIVE),
                )
            ]
        )
        fig_docs.update_layout(
            template=PLOTLY_TEMPLATE,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=max(280, 32 * len(document_df)),
            xaxis_title="Negative feedback count",
            yaxis_title=None,
        )
        st.plotly_chart(fig_docs, use_container_width=True)

        with st.expander("View as table"):
            st.dataframe(document_df, use_container_width=True, hide_index=True)

st.divider()

# --------------------------------------------------
# MOST COMMON QUESTIONS
# --------------------------------------------------

st.subheader("❓ Most Common Questions")

question_counts = df["question"].value_counts().head(10).sort_values(ascending=True)

if question_counts.empty:
    st.caption("No question data available for the selected filters.")
else:
    fig_bar = go.Figure(
        data=[
            go.Bar(
                x=question_counts.values,
                y=question_counts.index,
                orientation="h",
                marker=dict(color=COLOR_ACCENT_2),
            )
        ]
    )
    fig_bar.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(320, 32 * len(question_counts)),
        xaxis_title="Times asked",
        yaxis_title=None,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# --------------------------------------------------
# RESPONSE TIME TREND
# --------------------------------------------------

if "response_time" in df.columns and df["response_time"].notna().any():
    st.subheader("⏱️ Response Time Trend")

    daily_response_time = (
        df.dropna(subset=["response_time"])
        .groupby("date")["response_time"]
        .mean()
        .reset_index()
    )

    fig_response = go.Figure(
        data=[
            go.Scatter(
                x=daily_response_time["date"],
                y=daily_response_time["response_time"],
                mode="lines+markers",
                line=dict(color=COLOR_ACCENT_1, width=2.5),
                marker=dict(size=6),
                name="Avg response time",
            )
        ]
    )
    fig_response.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        xaxis_title=None,
        yaxis_title="Seconds",
    )
    st.plotly_chart(fig_response, use_container_width=True)

    st.divider()

    # ------------------------------------------
    # SLOWEST / FASTEST QUESTIONS (side by side)
    # ------------------------------------------

    slow_col, fast_col = st.columns(2)

    timing_column_config = {
        "question": st.column_config.TextColumn("Question", width="large"),
        "technology": st.column_config.TextColumn("Technology"),
        "response_time": st.column_config.NumberColumn("Response Time (s)", format="%.2f s"),
        "feedback": st.column_config.TextColumn("Feedback"),
    }

    with slow_col:
        st.subheader("🐢 Slowest Questions")

        slowest_columns = ["question", "technology", "response_time", "feedback"]
        slowest_questions = (
            df.dropna(subset=["response_time"])
            .sort_values("response_time", ascending=False)
            .head(5)[slowest_columns]
            .reset_index(drop=True)
        )
        slowest_questions["feedback"] = slowest_questions["feedback"].map(
            {"positive": "👍 Positive", "negative": "👎 Negative"}
        )

        st.dataframe(
            slowest_questions,
            use_container_width=True,
            hide_index=True,
            column_config=timing_column_config,
        )

    with fast_col:
        st.subheader("⚡ Fastest Questions")

        fastest_columns = ["question", "technology", "response_time", "feedback"]
        fastest_questions = (
            df.dropna(subset=["response_time"])
            .sort_values("response_time", ascending=True)
            .head(5)[fastest_columns]
            .reset_index(drop=True)
        )
        fastest_questions["feedback"] = fastest_questions["feedback"].map(
            {"positive": "👍 Positive", "negative": "👎 Negative"}
        )

        st.dataframe(
            fastest_questions,
            use_container_width=True,
            hide_index=True,
            column_config=timing_column_config,
        )

    st.divider()

# --------------------------------------------------
# RECENT FEEDBACK
# --------------------------------------------------

st.subheader("📝 Recent Feedback")

display_columns = ["timestamp", "question", "rewritten_question", "feedback"]
column_config = {
    "timestamp": st.column_config.DatetimeColumn("Time", format="D MMM YYYY, HH:mm"),
    "question": st.column_config.TextColumn("Question", width="medium"),
    "rewritten_question": st.column_config.TextColumn("Rewritten Query", width="medium"),
    "feedback": st.column_config.TextColumn("Feedback"),
}

if "technology" in df.columns and df["technology"].notna().any():
    display_columns.append("technology")
    column_config["technology"] = st.column_config.TextColumn("Technology")

if "response_time" in df.columns:
    display_columns.append("response_time")
    column_config["response_time"] = st.column_config.NumberColumn("Response Time (s)", format="%.2f s")

if "retrieved_document_count" in df.columns:
    display_columns.append("retrieved_document_count")
    column_config["retrieved_document_count"] = st.column_config.NumberColumn("Docs Retrieved")

stage_column_labels = {
    "query_rewriting_time": "Rewrite (s)",
    "retrieval_time": "Retrieval (s)",
    "reranking_time": "Reranking (s)",
    "llm_generation_time": "LLM Gen (s)",
}

for column, label in stage_column_labels.items():
    if column in df.columns and df[column].notna().any():
        display_columns.append(column)
        column_config[column] = st.column_config.NumberColumn(label, format="%.2f s")

if "retrieved_document_titles" in df.columns and df["retrieved_document_titles"].notna().any():
    display_columns.append("retrieved_document_titles")
    column_config["retrieved_document_titles"] = st.column_config.ListColumn("Retrieved Docs")

recent = df[display_columns].sort_values("timestamp", ascending=False).reset_index(drop=True)
recent["feedback"] = recent["feedback"].map({"positive": "👍 Positive", "negative": "👎 Negative"})

st.dataframe(
    recent,
    use_container_width=True,
    hide_index=True,
    column_config=column_config,
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption(f"🤖 {APP_NAME} \u2014 RAG Monitoring and Feedback Dashboard")