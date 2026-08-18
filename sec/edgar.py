# sec_agent.py
import os, json, requests
from dotenv import load_dotenv
from groq import Groq
from rag.db import load_chat

load_dotenv()

client      = Groq(api_key=os.getenv("GROQ_API_KEY"))
SEC_HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT")}

# ── SEC Tool Functions ────────────────────────────────────

def ticker_to_cik(ticker: str) -> str:
    data = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS
    ).json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"])
    raise ValueError(f"Ticker {ticker} not found")


def fetch_sec_filings(ticker: str, form_type: str = "10-K", n: int = 3) -> list[dict]:
    cik    = ticker_to_cik(ticker)
    padded = cik.zfill(10)
    data   = requests.get(
        f"https://data.sec.gov/submissions/CIK{padded}.json",
        headers=SEC_HEADERS
    ).json()

    filings = data["filings"]["recent"]
    results = []

    for i, form in enumerate(filings["form"]):
        if form == form_type and len(results) < n:
            accession = filings["accessionNumber"][i].replace("-", "")
            doc_name  = filings["primaryDocument"][i]
            filed_at  = filings["filingDate"][i]
            cik_raw   = cik.lstrip("0") or cik

            results.append({
                "filed_at":  filed_at,
                "form_type": form_type,
                "url": f"https://www.sec.gov/Archives/edgar/data/{cik_raw}/{accession}/{doc_name}",
                "filename": f"{ticker}_{form_type}_{filed_at}.htm",
            })

    return results


def download_doc(url: str, filename: str, save_dir: str = "edgar_docs") -> str:
    """
    Download an SEC filing if it doesn't already exist locally.
    Returns the local file path.
    """
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, filename)

    if not os.path.exists(path):
        r = requests.get(url, headers=SEC_HEADERS)
        r.raise_for_status()

        with open(path, "wb") as f:
            f.write(r.content)

    return path




from html.parser import HTMLParser


def extract_text(path: str) -> str:
    """
    Extract plain text from an SEC HTML filing.
    Returns the complete text.
    """

    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text = []

        def handle_data(self, data):
            data = data.strip()
            if data:
                self.text.append(data)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    parser = TextExtractor()
    parser.feed(html)

    return " ".join(parser.text)