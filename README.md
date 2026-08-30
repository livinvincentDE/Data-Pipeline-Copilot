<div align="center">

# 🤖 Data Pipeline Copilot

### An end-to-end Retrieval-Augmented Generation (RAG) assistant for troubleshooting data engineering problems

*Hybrid retrieval · Cross-encoder reranking · LLM query rewriting · Evaluation & regression testing · CI/CD*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036?style=for-the-badge&logo=lightning&logoColor=white)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](#-license)
[![Build](https://img.shields.io/badge/CI%2FCD-passing-brightgreen?style=for-the-badge&logo=githubactions&logoColor=white)](#-cicd)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-blueviolet?style=for-the-badge)](#-contributing)

</div>

---

## 📖 Table of Contents

- [🚀 Overview](#-overview)
- [✨ Features](#-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🧰 Tech Stack](#-tech-stack)
- [📂 Project Structure](#-project-structure)
- [⚙️ Getting Started](#️-getting-started)
- [🖥️ Usage](#️-usage)
- [📊 Monitoring Dashboard](#-monitoring-dashboard)
- [🧪 Evaluation & Regression Testing](#-evaluation--regression-testing)
- [🔄 CI/CD](#-cicd)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🚀 Overview

**Data Pipeline Copilot** is an intelligent RAG-powered assistant built to help data engineers troubleshoot the problems they hit every day, across:

| Technology | Focus Areas |
|---|---|
| ⚡ **Apache Spark** | job tuning, skew, spilling, partitioning |
| 🌬️ **Apache Airflow** | DAG failures, scheduling, retries |
| 📩 **Apache Kafka** | consumer lag, rebalancing, throughput |
| 🔧 **dbt** | model errors, testing, incremental builds |
| ✅ **Data Quality** | validation, checks, anomaly detection |

Instead of leaning on an LLM alone, this project implements a **production-style RAG pipeline** — hybrid retrieval, reranking, evaluation, and regression detection — so answers stay grounded, measurable, and trustworthy over time.

---

## ✨ Features

- 🔍 **Hybrid Retrieval** — BM25 keyword search + dense vector search combined
- 🎯 **Cross-Encoder Reranking** — reorders candidates for precision before generation
- ✍️ **LLM Query Rewriting** — clarifies vague or shorthand questions before retrieval
- 💬 **LLM Answer Generation** — grounded, context-aware responses via Groq
- 📈 **Retrieval Evaluation** — measurable retrieval quality, not vibes
- 🧪 **Regression Testing** — catches retrieval/answer quality regressions automatically
- 📊 **Baseline Comparison** — track improvements (or regressions) release over release
- 👍👎 **User Feedback Loop** — thumbs up/down feeds directly into monitoring
- ⏱️ **Per-Stage Latency Tracking** — rewriting, retrieval, reranking, and generation timed separately
- 🐢 **Automatic Bottleneck Detection** — flags whichever pipeline stage is slowing things down
- 🧠 **Answer Quality by Technology** — see which technology's answers need the most work
- 🔬 **Negative Feedback Investigation** — root-cause view of what went wrong and which documents were involved
- 🤖 **Automated Testing & CI/CD** — every change is validated before it ships

---

## 🏗️ System Architecture

```text
                        ┌──────────────────┐
                        │   User Question   │
                        └────────┬──────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │    Query Rewriting      │
                    │        Groq LLM         │
                    └───────────┬─────────────┘
                                │
                                ▼
              ┌──────────────────────────────────┐
              │          Hybrid Retrieval         │
              │                                    │
              │    BM25 Search + Vector Search     │
              └────────────────┬───────────────────┘
                               │
                               ▼
                   ┌───────────────────────┐
                   │   Cross-Encoder        │
                   │      Reranking         │
                   └───────────┬────────────┘
                               │
                               ▼
                   ┌───────────────────────┐
                   │  Relevant Documents    │
                   └───────────┬────────────┘
                               │
                               ▼
                   ┌───────────────────────┐
                   │  Context Construction  │
                   └───────────┬────────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │      Groq LLM       │
                     │  Answer Generation  │
                     └──────────┬──────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │  Final Answer  │
                        └───────┬────────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │  👍 / 👎 Feedback    │
                     │  → Monitoring       │
                     └────────────────────┘
```

---

## 📸 Screenshots

![Copilot answering a Spark troubleshooting question](docs/screenshots/copilot-answer.png)

![Monitoring dashboard overview](docs/screenshots/dashboard-overview.png)

![Monitoring dashboard feedback trends](docs/screenshots/dashboard-feedback-trends.png)

---

## 🧰 Tech Stack

<div align="left">

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Groq](https://img.shields.io/badge/-Groq-F55036?style=flat-square&logo=lightning&logoColor=white)
![Azure](https://img.shields.io/badge/-Azure-0078D4?style=flat-square&logo=microsoftazure&logoColor=white)
![Apache Spark](https://img.shields.io/badge/-Apache%20Spark-E25A1C?style=flat-square&logo=apachespark&logoColor=white)
![Apache Airflow](https://img.shields.io/badge/-Apache%20Airflow-017CEE?style=flat-square&logo=apacheairflow&logoColor=white)
![Apache Kafka](https://img.shields.io/badge/-Apache%20Kafka-231F20?style=flat-square&logo=apachekafka&logoColor=white)
![dbt](https://img.shields.io/badge/-dbt-FF694B?style=flat-square&logo=dbt&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/-GitHub%20Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white)
![Pytest](https://img.shields.io/badge/-Pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)

</div>

| Layer | Technology |
|---|---|
| **UI / App** | Streamlit |
| **LLM** | Groq (query rewriting + answer generation) |
| **Retrieval** | BM25 + vector search (hybrid) |
| **Reranking** | Cross-encoder model |
| **Data / Analytics** | Pandas, Plotly |
| **Monitoring** | Custom feedback & observability dashboard |
| **Testing** | Pytest, retrieval/answer regression suite |
| **CI/CD** | GitHub Actions |

---

## 📂 Project Structure

```text
data-pipeline-copilot/
├── app/
│   ├── rag.py              # Core RAG pipeline (rewrite → retrieve → rerank → generate)
│   ├── feedback.py         # save_feedback() / load_feedback()
│   └── dashboard.py        # 📊 Monitoring dashboard (Streamlit page)
├── streamlit_app.py         # 🤖 Main Copilot chat UI
├── evaluation/
│   ├── regression_tests/    # Baseline comparison & regression detection
│   └── metrics.py           # Retrieval evaluation metrics
├── data/                     # Knowledge base source documents
├── tests/                    # Unit & integration tests
├── .github/workflows/        # CI/CD pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/data-pipeline-copilot.git
cd data-pipeline-copilot

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🖥️ Usage

Launch the Copilot chat interface:

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501` and start asking questions like:

> *"Why is my Spark job spilling to disk?"*
> *"My Airflow DAG is stuck in a queued state — what should I check?"*
> *"How do I fix consumer lag in Kafka?"*

---

## 📊 Monitoring Dashboard

Launch the observability dashboard (Streamlit multipage or standalone):

```bash
streamlit run app/dashboard.py
```

The dashboard tracks:

- 📌 Feedback summary (total, positive, negative, positive rate)
- ⚡ Overall RAG performance (avg response time, avg documents retrieved)
- 🧠 Answer quality by technology, with an auto-flagged **technology needing attention**
- ⏱️ Per-stage pipeline latency (query rewriting → retrieval → reranking → generation)
- 🐢 Automatic bottleneck detection with severity-based alerts
- 🔍 Negative feedback investigation, including which documents were involved
- ❓ Most common questions & 🐢 slowest / ⚡ fastest questions

---

## 🧪 Evaluation & Regression Testing

Every change to the retrieval or generation pipeline is checked against a fixed evaluation set before merging:

```bash
pytest evaluation/regression_tests/
```

This compares current retrieval/answer quality against a stored **baseline**, flagging any regression before it reaches production.

---

## 🔄 CI/CD

GitHub Actions runs on every push and pull request:

- ✅ Unit & integration tests
- ✅ Retrieval regression suite
- ✅ Lint / formatting checks

See [`.github/workflows/`](.github/workflows/) for the pipeline definition.

---

## 🗺️ Roadmap

- [ ] Multi-turn conversational memory
- [ ] Support for additional data stack technologies (Snowflake, Databricks)
- [ ] Vector store swap-in support (Azure AI Search / pgvector / Pinecone)
- [ ] Slack / Teams bot integration
- [ ] Fine-grained per-user analytics

---

## 🤝 Contributing

Contributions are welcome! 🎉

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for data engineers who are tired of Googling the same Spark error twice.**

⭐ If this project helped you, consider giving it a star!

</div>
