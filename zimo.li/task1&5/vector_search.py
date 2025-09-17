import lancedb
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import multiprocessing as mp
import numpy as np
from lancedb.pydantic import LanceModel, Vector
from typing import List, Dict
import voyageai
from openai import OpenAI
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dotenv import load_dotenv
import tiktoken
import logging

# --- Configuration ---
DB_PATH = "/home/ubuntu/database"
TABLE_NAME = "task5v3"
VOYAGEAI_MODEL_ID = "voyage-3.5-lite"
OPENAI_MODEL_ID = "text-embedding-3-small"
EMBEDDING_DIM = 1024

db = lancedb.connect(DB_PATH)
table = db.open_table(TABLE_NAME)


load_dotenv()
voyageai_api_key = os.environ.get("VOYAGE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
query = "machine learning"
voyage_result = voyageai.Client().embed(query, model=VOYAGEAI_MODEL_ID, truncation=True, output_dimension=EMBEDDING_DIM)
voyage_query = voyage_result.embeddings[0]
openai_result = OpenAI().embeddings.create(input=query, model=OPENAI_MODEL_ID, dimensions=EMBEDDING_DIM)
openai_query = openai_result.data[0].embedding
print(f"VoyageAI Embedding: {voyage_query[:10]}")
print(f"OpenAI Embedding: {openai_query[:10]}")


# 替换 tbl -> table
table.create_index()
v_result = table.search(voyage_query).limit(1).to_pandas()
o_result = table.search(openai_query).limit(1).to_pandas()

print("VoyageAI Search Result:")
print(v_result["title"])
print(v_result["vector_title_voyageai"].iloc[0][:10])
print(v_result["vector_title_openai"].iloc[0][:10])
print()

print("OpenAI Search Result:")
print(o_result["title"])
print(o_result["vector_title_voyageai"].iloc[0][:10])
print(o_result["vector_title_openai"].iloc[0][:10])
