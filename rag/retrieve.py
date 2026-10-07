# for fetch related chunks from db according to question

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from groq import Groq
import os
from dotenv import load_dotenv

from rag.db import related_chunks


# def get_retrieve(user_id, question, k=1):

#     result = related_chunks(user_id, question, k=1)

#     releted_context = "\n".join([row[0] for row in result])

#     return releted_context



def get_retrieve(user_id, question, k=4, document_ids=None, debug=True):
    result = related_chunks(user_id, question, k=k, document_ids=document_ids)

    if debug:
        print(f"\n{'=' * 70}")
        print(f"QUESTION: {question}")
        print(f"RETRIEVED {len(result)} chunks")
        print("=" * 70)
        for rank, (content, page, distance) in enumerate(result, 1):
            # distance is cosine distance (<=>): lower = closer
            # 1.0 means the chunk came only from keyword search (no vector hit)
            similarity = 1 - distance
            print(f"\n[{rank}] page={page} | distance={distance:.4f} | similarity={similarity:.4f}")
            print(content[:600].replace("\n", " "))
            print("-" * 70)

    related_context = "\n".join(row[0] for row in result)
    return related_context