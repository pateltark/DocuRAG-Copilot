# for fetch related chunks from db according to question

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from groq import Groq
import os
from dotenv import load_dotenv

from rag.db import related_chunks


def get_retrieve(user_id, question, k=1):

    result = related_chunks(user_id, question, k=1)

    releted_context = "\n".join([row[0] for row in result])

    return releted_context