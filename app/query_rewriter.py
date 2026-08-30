import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# Find project root
BASE_DIR = Path(__file__).resolve().parent.parent


# Load .env
load_dotenv(
    BASE_DIR / ".env",
    override=True
)


# Get Groq API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. "
        "Check your .env file."
    )


# Create Groq client
client = Groq(api_key=api_key)


def rewrite_query(question):
    """
    Rewrite a user's question into a clearer
    search query for the data engineering
    knowledge base.
    """

    prompt = f"""
You are a query rewriting assistant for a data engineering
troubleshooting knowledge base.

Rewrite the user's question into a clear and specific search query.

Rules:
- Preserve the original meaning.
- Do not answer the question.
- Do not add information that the user did not provide.
- Return ONLY the rewritten query.
- Keep the rewritten query concise.

User question:
{question}

Rewritten query:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    rewritten_query = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    return rewritten_query


if __name__ == "__main__":

    test_questions = [

        "spark is slow help",

        "my dag broken",

        "kafka behind",

        "why duplicate data"
    ]

    for question in test_questions:

        rewritten = rewrite_query(question)

        print("\n" + "=" * 60)

        print("Original Query:")

        print(question)

        print("\nRewritten Query:")

        print(rewritten)