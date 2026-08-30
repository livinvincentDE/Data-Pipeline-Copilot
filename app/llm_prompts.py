def build_basic_prompt(question, context):
    """
    Simple RAG prompt.
    """

    return f"""
Answer the user's question using the context below.

Context:
{context}

Question:
{question}

Answer:
"""


def build_improved_prompt(question, context):
    """
    Structured and grounded RAG prompt.
    """

    return f"""
You are Data Pipeline Copilot, an AI assistant that helps
data engineers troubleshoot data pipeline problems.

Answer the user's question using ONLY the information
provided in the retrieved context.

Rules:

- Do not invent information.
- Do not use knowledge outside the context.
- If the context does not contain enough information,
  clearly say so.
- Give practical troubleshooting guidance.
- Use clear bullet points when helpful.
- Keep the answer concise and useful.

Retrieved Context:

{context}

User Question:

{question}

Provide a helpful troubleshooting answer:
"""