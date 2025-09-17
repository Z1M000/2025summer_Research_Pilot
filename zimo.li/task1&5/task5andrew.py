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
logging.basicConfig(
    filename="/home/ubuntu/openalex/workspace/zimo.li/task5.log",               
    level=logging.INFO,                
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# --- Pydantic Schema for LanceDB ---
class Paper(LanceModel):
    openalex_id: str
    doi: str
    publication_date: str
    title: str
    abstract: str
    vector_title_voyageai: Vector(EMBEDDING_DIM)
    vector_abstract_voyageai: Vector(EMBEDDING_DIM)
    vector_title_openai: Vector(EMBEDDING_DIM)
    vector_abstract_openai: Vector(EMBEDDING_DIM)
    mag_id: str
    pmid: str
    pmcid: str
    arxiv_id: str
    raw_affiliation_strings: List[str]
    institution_ids: List[str]
    authors_id: List[str]
    authors_name: List[str]
    orcid: List[str]
    referenced_works: List[str]
    topics: List[str]
    cited_by_api_url: str
    citations: List[str]


def count_tokens(batch_texts: List[str], provider: str):
    """Tokenize a batch of texts and return their lengths."""
    if provider == "voyageai":
        tokenizer = AutoTokenizer.from_pretrained(f"voyageai/{VOYAGEAI_MODEL_ID}")
        return [len(encoded) for encoded in tokenizer.batch_encode_plus(batch_texts, add_special_tokens=False)["input_ids"]]
    elif provider == "openai":
        tokenizer = tiktoken.encoding_for_model(OPENAI_MODEL_ID)
        return [len(tokenizer.encode(text)) for text in batch_texts]
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def pack_batches(texts_df: pd.DataFrame, max_tokens_per_batch: int, max_inputs_per_batch: int):
    """Pack texts into batches based on token count and input limits."""
    batches = []
    current_batch = []
    current_tokens = 0

    for _, row in texts_df.iterrows():
        if (current_tokens + row["num_tokens"] > max_tokens_per_batch or len(current_batch) >= max_inputs_per_batch) and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(row["text"])
        current_tokens += row["num_tokens"]

    if current_batch:
        batches.append(current_batch)

    return batches


def get_embeddings_in_batches(voyageai_client: voyageai.Client, openai_client: OpenAI, texts: List[str]) -> Dict[str, Dict[str, list]]:
    """Get embeddings for a list of texts using separate batching logic for each API."""
    if not texts:
        return {"voyageai": {}, "openai": {}}

    # Separate token counting for each API
    token_counts_voyageai = count_tokens(texts, "voyageai")
    token_counts_openai = count_tokens(texts, "openai")

    texts_df_voyageai = pd.DataFrame({"text": texts, "num_tokens": token_counts_voyageai}).sort_values("num_tokens", ascending=False)
    texts_df_openai = pd.DataFrame({"text": texts, "num_tokens": token_counts_openai}).sort_values("num_tokens", ascending=False)

    batches_voyageai = pack_batches(texts_df_voyageai, max_tokens_per_batch=1000000, max_inputs_per_batch=1000)
    batches_openai = pack_batches(texts_df_openai, max_tokens_per_batch=300000, max_inputs_per_batch=2048)

    all_embeddings = {"voyageai": {}, "openai": {}}

    # Get VoyageAI embeddings
    for batch in batches_voyageai:
        try:
            result = voyageai_client.embed(batch, model=VOYAGEAI_MODEL_ID, truncation=True, output_dimension=EMBEDDING_DIM)
            for text, embedding in zip(batch, result.embeddings):
                all_embeddings["voyageai"][text] = embedding
        except Exception as e:
            print(f"Error embedding VoyageAI batch: {e}. Skipping.")

    # Get OpenAI embeddings
    for batch in batches_openai:
        try:
            result = openai_client.embeddings.create(input=batch, model=OPENAI_MODEL_ID, dimensions=EMBEDDING_DIM)
            for text, embedding_obj in zip(batch, result.data):
                all_embeddings["openai"][text] = embedding_obj.embedding
        except Exception as e:
            print(f"Error embedding OpenAI batch: {e}. Skipping.")

    return all_embeddings


# --- Main Logic ---


def works2lancedb(input_path: str, voyageai_api_key: str, openai_api_key: str):
    voyageai_client = voyageai.Client(api_key=voyageai_api_key)
    openai_client = OpenAI(api_key=openai_api_key)
    """Initializes the database from a JSONL file using robust, optimized embedding."""

    # 1. Load and de-duplicate new data in chunks to handle large files
    try:
        chunk_size = 10000  # Process 10,000 lines at a time
        chunks = pd.read_json(input_path, lines=True, chunksize=chunk_size)
    except Exception as e:
        print(f"Error reading JSON file {input_path}: {e}")
        return

    df = pd.concat([chunk.drop_duplicates(subset=["id"], keep="last") for chunk in chunks], ignore_index=True)
    del chunks
    # 2. Collect all unique, non-empty texts for embedding
    titles = [t for t in df["title"].dropna().unique() if t]
    abstracts = [a for a in df["abstract"].dropna().unique() if a]

    # 3. Get all embeddings using the new batching logic
    try:
        title_embeddings = get_embeddings_in_batches(voyageai_client, openai_client, titles)
    except Exception as e:
        print(f"Error generating title embeddings for file {input_path}: {e}")
        title_embeddings = {"voyageai": {}, "openai": {}}

    try:
        abstract_embeddings = get_embeddings_in_batches(voyageai_client, openai_client, abstracts)
    except Exception as e:
        print(f"Error generating abstract embeddings for file {input_path}: {e}")
        abstract_embeddings = {"voyageai": {}, "openai": {}}

    del titles, abstracts, voyageai_client, openai_client

    # Create a new, clean DataFrame to prevent corruption
    final_df = pd.DataFrame()

    # Rename 'id' to 'openalex_id' to match the schema
    df.rename(columns={"id": "openalex_id"}, inplace=True)

    # Handle nested 'ids' columns safely by adding them to the main df
    df["ids"] = df["ids"].apply(lambda x: {k: str(v) for k, v in x.items()} if isinstance(x, dict) else {})
    ids_df = pd.json_normalize(df["ids"].fillna({}))
    for col_name, source_key in [("mag_id", "mag"), ("pmid", "pmid"), ("pmcid", "pmcid"), ("arxiv_id", "arxiv")]:
        if source_key in ids_df.columns:
            df[col_name] = ids_df[source_key]
        # If the key is missing for all rows, the column won't exist, so we handle it later

    # Ensure all specified string columns are present and correctly typed
    string_cols = ["openalex_id", "doi", "publication_date", "title", "abstract", "mag_id", "pmid", "pmcid", "arxiv_id", "cited_by_api_url"]
    for col in string_cols:
        if col not in df.columns:
            df[col] = ""  # Add column if it's completely missing
        final_df[col] = df[col].fillna("").astype(str)

    # Handle list-of-string columns safely
    authors_series = df["authors"].fillna(pd.Series([[] for _ in range(len(df))]))
    final_df["authors_id"] = authors_series.apply(lambda alist: [str(a.get("id", "")) for a in alist])
    final_df["authors_name"] = authors_series.apply(lambda alist: [str(a.get("display_name", "")) for a in alist])
    final_df["orcid"] = authors_series.apply(lambda alist: [str(a.get("orcid", "")) for a in alist])

    topics_series = df["topics"].fillna(pd.Series([[] for _ in range(len(df))]))
    final_df["topics"] = topics_series.apply(lambda tlist: [str(t.get("display_name", "")) for t in tlist])

    # Handle other list columns safely
    for col in ["raw_affiliation_strings", "institution_ids", "referenced_works", "citations"]:
        if col not in df.columns:
            final_df[col] = pd.Series([[] for _ in range(len(df))])
        else:
            final_df[col] = df[col].fillna(pd.Series([[] for _ in range(len(df))]))

    # Map embeddings
    final_df["vector_title_voyageai"] = df["title"].map(title_embeddings["voyageai"])
    final_df["vector_abstract_voyageai"] = df["abstract"].map(abstract_embeddings["voyageai"])
    final_df["vector_title_openai"] = df["title"].map(title_embeddings["openai"])
    final_df["vector_abstract_openai"] = df["abstract"].map(abstract_embeddings["openai"])

    # Fill missing embeddings with zeros
    final_df["vector_title_voyageai"] = final_df["vector_title_voyageai"].apply(lambda x: x if isinstance(x, (list, np.ndarray)) else np.zeros(EMBEDDING_DIM))
    final_df["vector_abstract_voyageai"] = final_df["vector_abstract_voyageai"].apply(lambda x: x if isinstance(x, (list, np.ndarray)) else np.zeros(EMBEDDING_DIM))
    final_df["vector_title_openai"] = final_df["vector_title_openai"].apply(lambda x: x if isinstance(x, (list, np.ndarray)) else np.zeros(EMBEDDING_DIM))
    final_df["vector_abstract_openai"] = final_df["vector_abstract_openai"].apply(lambda x: x if isinstance(x, (list, np.ndarray)) else np.zeros(EMBEDDING_DIM))

    db = lancedb.connect(DB_PATH)

    if TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)
        try:
            table.add(data=final_df)
            print(f"Successfully added {len(final_df)} papers to the database from file {input_path}.")
        except Exception as e:
            print(f"Error adding data to the database for file {input_path}: {e}")
    else:
        print(f"Creating new table '{TABLE_NAME}'...")
        table = db.create_table(TABLE_NAME, schema=Paper)
        table.add(data=final_df)

    print(f"Added {len(final_df)} papers to the database.")


def get_smallest_files(directory, extension=".jsonl"):
    files = []
    for path in Path(directory).rglob(f"*{extension}"):
        try:
            size = path.stat().st_size
            files.append((size, str(path)))
        except FileNotFoundError:
            continue
    files.sort()
    return [file for _, file in files]


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    load_dotenv()
    data_dir = "/home/ubuntu/openalex/workspace/zimo.li/task5v3files"
    smallest_files = get_smallest_files(data_dir)
    num_files = len(smallest_files)

    voyageai_api_key = os.environ.get("VOYAGE_API_KEY")
    if not voyageai_api_key:
        raise ValueError("VOYAGE_API_KEY environment variable not set")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    failed_files = []
    for i, file in enumerate(smallest_files):
        if not os.path.isfile(file) or os.path.getsize(file) == 0:
            print(f"Skipping empty or missing file: {file}")
            continue
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(works2lancedb, file, voyageai_api_key, openai_api_key)
            try:
                future.result()
                print(f"Successfully processed file: {file}")
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                failed_files.append(file)

        print(f"{num_files - (i +1)} files remaining | {((i + 1) / num_files):.2%} complete")
    if failed_files:
        print("\nRetrying failed files...")
        for file in failed_files:
            try:
                works2lancedb(file, voyageai_api_key, openai_api_key)
                print(f"Successfully retried file: {file}")
            except Exception as e:
                print(f"Retry failed for file {file}: {e}")
    print("\nDatabase initialization complete. Check logs for any errors.")

    db = lancedb.connect(DB_PATH)
    table = db.open_table(TABLE_NAME)
    pandas_df = table.to_pandas()
    print(pandas_df.head())
    pandas_df.to_csv("/home/ubuntu/openalex/workspace/zimo.li/view.csv", index=True)
