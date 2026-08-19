#planner.py

import sys
from pathlib import Path

# Add project root (intelligent_financial_research_copilot) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Literal

from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from rag.db import get_sec_document, save_sec_vector
from agent.tools import TOOLS
import json
import os


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ======================================================
# Models
# ======================================================

from typing import Literal

class DocumentRequest(BaseModel):
    company: str
    ticker: str | None = None
    form_type: Literal[
        "10-K",
        "10-Q",
        "8-K",
        "DEF 14A",
        "S-1"
    ] | None = None
    period: str | None = None


class PlannerOutput(BaseModel):

    action: Literal[
        "FETCH",
        "CURRENT_DOC"
    ]

    documents: list[DocumentRequest] = []


# ======================================================
# Prompt
# ======================================================

SYSTEM_PROMPT = """
You are an SEC routing agent.

Your ONLY job is to decide whether the user wants:

1. FETCH
   - The user is requesting a new SEC filing.
   - Examples:
        Show Tesla latest 10-K
        Download Apple's latest 8-K
        Microsoft quarterly report
        Compare Tesla and Apple revenue

2. CURRENT_DOC
   - The user is asking about the SEC filing that is already open.
   - Examples:
        What are the risks?
        Summarize this filing.
        What is on page 5?
        What did they say about revenue?
        Explain this document.

Rules

If action is FETCH:
- Extract every requested document.
- Populate documents.

If action is CURRENT_DOC:
- documents must be [].

Never invent ticker names.

Never use "Unknown".

If you cannot identify a company and the question is clearly referring to the current document,
return

{
  "action":"CURRENT_DOC",
  "documents":[]
}
"""


# ======================================================
# Planner
# ======================================================

def planner(question: str) -> PlannerOutput:
    # Append schema requirements directly to prompt
    json_schema_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "You MUST respond ONLY with a valid JSON object matching this schema:\n"
        f"{PlannerOutput.model_json_schema()}"
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b", # Note: use standard groq models like llama-3.3-70b
        temperature=0,
        response_format={"type": "json_object"},  # Enforces valid JSON without breaking on tool choices
        messages=[
            {
                "role": "system",
                "content": json_schema_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    raw_json = response.choices[0].message.content

    print("\nLLM Output\n")
    print(raw_json)

    # Directly validate with Pydantic
    return PlannerOutput.model_validate_json(raw_json)




def check_avail_sec_doc(planner_output):

    results = []

    for doc in planner_output.documents:  # doc = ticker, form_type.. 

        db_row = get_sec_document(
            ticker=doc.ticker,
            form_type=doc.form_type
        )

        results.append({
            "request": doc,
            "db_row": db_row
        })

    return results

# ======================================================
# Test
# ======================================================

if __name__ == "__main__":
    test_query = "can u give me latest TESLA 10-k news ?"

    has_doc = check_avail_sec_doc(test_query)

    print(has_doc)