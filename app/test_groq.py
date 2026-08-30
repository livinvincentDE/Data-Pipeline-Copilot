import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

print("Project directory:", BASE_DIR)
print(".env exists:", ENV_PATH.exists())

load_dotenv(ENV_PATH, override=True)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found in the .env file."
    )

print("API key loaded:", True)
print("Key starts with gsk_:", api_key.startswith("gsk_"))
print("Key length:", len(api_key))


client = Groq(api_key=api_key)


response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "Explain Apache Airflow in one sentence."
        }
    ],
    temperature=0
)


print("\n🤖 Groq Response:\n")
print(response.choices[0].message.content)