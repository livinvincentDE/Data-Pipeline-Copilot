"""
Data Pipeline Copilot
----------------------
A RAG-powered troubleshooting assistant for data engineering issues
(Apache Spark, Airflow, Kafka, dbt, and data quality).
"""

import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

APP_NAME = "PipelineIQ"
APP_TAGLINE = "Data Engineering Jobs Knowledge Base"

st.set_page_config(
    page_title=f"{APP_NAME} · {APP_TAGLINE}",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
        /* Overall page */
        .main {
            padding-top: 1.5rem;
        }

        /* Header */
        .app-title {
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #818cf8 0%, #38bdf8 60%, #34d399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            display: inline-block;
        }
        .app-tagline {
            font-size: 0.98rem;
            font-weight: 600;
            color: #a5b4fc;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.4rem;
        }
        .app-subtitle {
            color: #9ca3af;
            font-size: 1.02rem;
            margin-bottom: 1.3rem;
        }

        /* Sidebar brand */
        .sidebar-brand {
            font-size: 1.15rem;
            font-weight: 800;
            background: linear-gradient(90deg, #818cf8 0%, #38bdf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.1rem;
        }

        /* Topic pills */
        .topic-pill {
            display: inline-block;
            background-color: #eef2ff;
            color: #4338ca;
            border-radius: 999px;
            padding: 0.25rem 0.75rem;
            margin: 0.15rem;
            font-size: 0.82rem;
            font-weight: 500;
        }

        /* Source chip */
        .source-chip {
            background-color: rgba(99, 102, 241, 0.08);
            color: inherit;
            border-radius: 8px;
            padding: 0.6rem 0.9rem;
            margin-bottom: 0.5rem;
            font-size: 0.88rem;
            border-left: 3px solid #6366f1;
        }

        /* Metadata pill (technology / topic / doc id) */
        .meta-pill {
            display: inline-block;
            background-color: rgba(56, 189, 248, 0.12);
            color: #38bdf8;
            border-radius: 6px;
            padding: 0.15rem 0.55rem;
            margin: 0.1rem 0.3rem 0.1rem 0;
            font-size: 0.78rem;
            font-weight: 600;
        }

        /* Score badge */
        .score-badge {
            display: inline-block;
            background-color: rgba(52, 211, 153, 0.12);
            color: #34d399;
            border-radius: 6px;
            padding: 0.15rem 0.55rem;
            margin: 0.1rem 0.3rem 0.1rem 0;
            font-size: 0.78rem;
            font-weight: 600;
            font-family: monospace;
        }

        /* History item */
        .history-question {
            font-weight: 600;
            font-size: 0.9rem;
        }
        .history-time {
            color: #9ca3af;
            font-size: 0.75rem;
        }

        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: question, answer, rewritten_question, documents, timestamp

if "last_result" not in st.session_state:
    st.session_state.last_result = None  # holds the most recent Q&A so it survives feedback-button reruns

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = None  # "positive" / "negative" for the current last_result

if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

TOPICS = [
    "Spark job tuning",
    "Airflow DAG failures",
    "Kafka consumer lag",
    "dbt model errors",
    "Data quality checks",
    "Partitioning strategy",
]

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown(f'<div class="sidebar-brand">🧠 {APP_NAME}</div>', unsafe_allow_html=True)
    st.caption(f"{APP_TAGLINE} — Spark, Airflow, Kafka & dbt troubleshooting")

    st.divider()

    st.markdown("**Quick topics**")
    for topic in TOPICS:
        if st.button(topic, use_container_width=True, key=f"topic_{topic}"):
            st.session_state.pending_question = f"Why is my {topic.lower()} happening / how do I fix it?"

    st.divider()

    st.markdown("**Session history**")
    if not st.session_state.history:
        st.caption("No questions asked yet.")
    else:
        for item in reversed(st.session_state.history[-8:]):
            st.markdown(
                f"""
                <div class="history-question">{item['question']}</div>
                <div class="history-time">{item['timestamp']}</div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("&nbsp;", unsafe_allow_html=True)

        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(f'<div class="app-tagline">🧠 {APP_TAGLINE}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="app-title">{APP_NAME}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Your go-to assistant for troubleshooting '
    "Spark, Airflow, Kafka, dbt, and data quality issues in production pipelines.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    " ".join(f'<span class="topic-pill">{t}</span>' for t in TOPICS),
    unsafe_allow_html=True,
)

st.divider()

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

col_input, col_button = st.columns([5, 1])

with col_input:
    question = st.text_input(
        "Ask a data engineering question",
        value=st.session_state.pending_question,
        placeholder="Example: Why is my Spark job spilling to disk?",
        label_visibility="collapsed",
    )

with col_button:
    ask_clicked = st.button("🚀 Ask", type="primary", use_container_width=True)

# Reset the pending question once it's been consumed by the input widget
st.session_state.pending_question = ""

# --------------------------------------------------
# ASK COPILOT
# --------------------------------------------------

if ask_clicked:

    if not question.strip():
        st.warning("Please enter a question before asking the copilot.")

    else:
        try:
            with st.spinner("🤖 Analyzing your pipeline issue..."):
                from app.rag import answer_question

                start_time = perf_counter()
                result = answer_question(question)
                elapsed_time = perf_counter() - start_time

            answer = result.get("answer", "No answer was returned.")
            rewritten_question = result.get("rewritten_question", "Not available")
            documents = result.get("documents", [])
            timestamp = datetime.now().strftime("%H:%M:%S")

            # Prefer a response_time reported by the RAG pipeline itself
            # (e.g. server-side timing that excludes network overhead);
            # fall back to the client-measured wall-clock time.
            response_time = result.get("response_time", elapsed_time)
            retrieved_document_count = result.get("retrieved_document_count", len(documents))

            # Persist as the "current" result so it survives reruns
            # triggered by the feedback buttons below.
            st.session_state.last_result = {
                "original_question": question,
                "answer": answer,
                "rewritten_question": rewritten_question,
                "documents": documents,
                "timestamp": timestamp,
                "response_time": response_time,
                "retrieved_document_count": retrieved_document_count,
            }
            st.session_state.feedback_given = None

            # Save to session history
            st.session_state.history.append(
                {
                    "question": question,
                    "answer": answer,
                    "rewritten_question": rewritten_question,
                    "documents": documents,
                    "timestamp": timestamp,
                    "response_time": response_time,
                    "retrieved_document_count": retrieved_document_count,
                }
            )

        except Exception as error:
            st.session_state.last_result = None
            st.error("Something went wrong while running the RAG pipeline.")
            with st.expander("Show error details"):
                st.exception(error)

# --------------------------------------------------
# RESULT DISPLAY (Answer / Rewritten Query / Retrieved Knowledge / Feedback)
# --------------------------------------------------
# Rendered from session_state rather than inline in the "ask_clicked" branch
# above, so the feedback buttons' own click-triggered rerun doesn't wipe
# the answer off the screen.

if st.session_state.last_result:
    result = st.session_state.last_result

    # ------------------------------------------
    # ANSWER
    # ------------------------------------------

    st.subheader("💡 Answer")
    with st.container(border=True):
        # Plain st.markdown (no raw HTML wrapper) so Streamlit's
        # theme-aware styling and native markdown parsing (lists,
        # code blocks, bold, etc.) render correctly in both
        # light and dark mode.
        st.markdown(result["answer"])

    st.caption(
        f"⏱️ Answered in {result['response_time']:.2f}s · "
        f"📄 {result['retrieved_document_count']} document(s) retrieved"
    )

    # ------------------------------------------
    # REWRITTEN SEARCH QUERY
    # ------------------------------------------

    with st.expander("✨ Rewritten search query"):
        st.code(result["rewritten_question"], language=None)

    # ------------------------------------------
    # RETRIEVED KNOWLEDGE
    # ------------------------------------------

    documents = result["documents"]
    st.subheader(f"📚 Retrieved knowledge ({len(documents)})")

    if documents:
        for index, item in enumerate(documents, start=1):
            document = item.get("document", {})
            title = document.get("title", "Unknown Document")

            with st.expander(f"{index}. {title}"):

                # Metadata pills
                st.markdown(
                    f'<span class="meta-pill">🔧 {document.get("technology", "N/A")}</span>'
                    f'<span class="meta-pill">🏷️ {document.get("topic", "N/A")}</span>'
                    f'<span class="meta-pill">🆔 {document.get("id", "N/A")}</span>',
                    unsafe_allow_html=True,
                )

                # Score badges
                score_html = ""
                if "score" in item:
                    score_html += f'<span class="score-badge">Retrieval: {item["score"]:.4f}</span>'
                if "reranker_score" in item:
                    score_html += (
                        f'<span class="score-badge">Reranker: {item["reranker_score"]:.4f}</span>'
                    )
                if score_html:
                    st.markdown(score_html, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("**Content**")
                st.markdown(document.get("content", ""))
    else:
        st.info("No retrieved documents available for this question.")

    # ------------------------------------------
    # USER FEEDBACK
    # ------------------------------------------

    st.divider()
    st.subheader("Was this answer helpful?")

    if st.session_state.feedback_given:
        verdict = "👍 Marked helpful" if st.session_state.feedback_given == "positive" else "👎 Marked not helpful"
        st.caption(f"{verdict} — thanks for the feedback on this answer.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Helpful", key=f"positive_{result['timestamp']}", use_container_width=True):
                try:
                    from app.feedback import save_feedback

                    save_feedback(
                        question=result["original_question"],
                        rewritten_question=result["rewritten_question"],
                        answer=result["answer"],
                        feedback="positive",
                        response_time=result.get("response_time"),
                        retrieved_document_count=result.get("retrieved_document_count"),
                    )
                    st.session_state.feedback_given = "positive"
                    st.success("Thank you for your feedback! 🎉")
                    st.rerun()
                except Exception as error:
                    st.error("Could not save feedback.")
                    st.exception(error)

        with col2:
            if st.button("👎 Not helpful", key=f"negative_{result['timestamp']}", use_container_width=True):
                try:
                    from app.feedback import save_feedback

                    save_feedback(
                        question=result["original_question"],
                        rewritten_question=result["rewritten_question"],
                        answer=result["answer"],
                        feedback="negative",
                        response_time=result.get("response_time"),
                        retrieved_document_count=result.get("retrieved_document_count"),
                    )
                    st.session_state.feedback_given = "negative"
                    st.info("Thank you! Your feedback helps improve the Copilot.")
                    st.rerun()
                except Exception as error:
                    st.error("Could not save feedback.")
                    st.exception(error)

elif not st.session_state.history:
    st.info(
        "👋 Pick a topic on the left, or type a question above to get started — "
        "e.g. *'Why is my Airflow DAG stuck in a queued state?'*"
    )