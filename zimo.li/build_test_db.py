import lancedb
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
import argparse
from lancedb.pydantic import LanceModel, Vector
from typing import List, Dict
import voyageai
from openai import OpenAI
from transformers import AutoTokenizer
from dotenv import load_dotenv
import tiktoken
import logging # (Zimo)

# --- Configuration ---
DB_PATH = "/home/ubuntu/database"
TABLE_NAME = "task11test_zimo"
VOYAGEAI_MODEL_ID = "voyage-3.5-lite"
OPENAI_MODEL_ID = "text-embedding-3-small"
EMBEDDING_DIM = 1024


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
            logging.warning(f"Error embedding VoyageAI batch: {e}. Skipping.")

    # Get OpenAI embeddings
    for batch in batches_openai:
        try:
            result = openai_client.embeddings.create(input=batch, model=OPENAI_MODEL_ID, dimensions=EMBEDDING_DIM)
            for text, embedding_obj in zip(batch, result.data):
                all_embeddings["openai"][text] = embedding_obj.embedding
        except Exception as e:
            logging.warning(f"Error embedding OpenAI batch: {e}. Skipping.")

    return all_embeddings


# --- Main Logic ---

def update_papers(input_path: str, api_key: str, openai_api_key: str):
    """Incrementally updates the database using robust, optimized embedding."""
    logging.info(f"Starting paper update process from file: {input_path}")

    # 1. Load and de-duplicate new data
    try:
        df = pd.read_json(input_path, lines=True)
        df.drop_duplicates(subset=["id"], keep="last", inplace=True)
    except Exception as e:
        logging.error(f"Error reading or parsing JSON file {input_path}: {e}")
        return

    if df.empty:
        logging.info("No papers in the input file. Exiting.")
        return

    df.rename(columns={"id": "openalex_id"}, inplace=True)

    # Connect to DB
    db = lancedb.connect(DB_PATH)
    table = None
    if TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)

    papers_to_process_df = df
    ids_to_delete = []

    if table:
        ids_in_input = df["openalex_id"].tolist()

        query_chunk_size = 500
        existing_papers_df_list = []
        logging.info("Checking for existing papers in the database...")
        for i in range(0, len(ids_in_input), query_chunk_size):
            id_chunk = ids_in_input[i : i + query_chunk_size]
            id_filter_str = ", ".join([f"'{_id}'" for _id in id_chunk])
            if id_filter_str:
                try:
                    results = table.search().where(f"openalex_id IN ({id_filter_str})").select(["openalex_id", "title", "abstract", "vector_title_voyageai", "vector_abstract_voyageai", "vector_title_openai", "vector_abstract_openai"]).to_pandas()
                    # Rename embedding columns to have _db suffix for merging
                    results = results.rename(columns={"vector_title_voyageai": "vector_title_voyageai_db", "vector_abstract_voyageai": "vector_abstract_voyageai_db", "vector_title_openai": "vector_title_openai_db", "vector_abstract_openai": "vector_abstract_openai_db"})
                    existing_papers_df_list.append(results)
                except Exception as e:
                    logging.warning(f"Warning: Could not query existing papers chunk. Assuming they are new. Error: {e}")

        if existing_papers_df_list:
            existing_papers_df = pd.concat(existing_papers_df_list, ignore_index=True)

            # Prepare for merge - handle NaNs
            df["title"] = df["title"].fillna("")
            df["abstract"] = df["abstract"].fillna("")
            existing_papers_df["title"] = existing_papers_df["title"].fillna("")
            existing_papers_df["abstract"] = existing_papers_df["abstract"].fillna("")

            merged_df = df.merge(existing_papers_df, on="openalex_id", how="left", suffixes=("", "_db"))

            is_new = merged_df["title_db"].isna()
            if "abstract_db" not in merged_df.columns:
                merged_df["abstract_db"] = np.nan

            # Check if title or abstract changed (for re-embedding decision)
            title_or_abstract_changed = ~is_new & ((merged_df["title"] != merged_df["title_db"]) | (merged_df["abstract"] != merged_df["abstract_db"]))

            # Split into different categories
            new_papers = merged_df[is_new].copy()
            changed_title_abstract = merged_df[title_or_abstract_changed].copy()
            unchanged_title_abstract = merged_df[~is_new & ~title_or_abstract_changed].copy()

            # All existing papers need to be updated (replaced with new data)
            papers_to_process_df = merged_df.copy()

            # Mark which ones need re-embedding vs copying embeddings
            papers_to_process_df["needs_embedding"] = is_new | title_or_abstract_changed

            # For papers that don't need re-embedding, copy over embeddings
            mask_copy_embeddings = ~papers_to_process_df["needs_embedding"]
            if mask_copy_embeddings.any():
                papers_to_process_df.loc[mask_copy_embeddings, "vector_title_voyageai"] = papers_to_process_df.loc[mask_copy_embeddings, "vector_title_voyageai_db"]
                papers_to_process_df.loc[mask_copy_embeddings, "vector_abstract_voyageai"] = papers_to_process_df.loc[mask_copy_embeddings, "vector_abstract_voyageai_db"]
                papers_to_process_df.loc[mask_copy_embeddings, "vector_title_openai"] = papers_to_process_df.loc[mask_copy_embeddings, "vector_title_openai_db"]
                papers_to_process_df.loc[mask_copy_embeddings, "vector_abstract_openai"] = papers_to_process_df.loc[mask_copy_embeddings, "vector_abstract_openai_db"]

            # All existing papers will be deleted and replaced
            ids_to_delete = merged_df[~is_new]["openalex_id"].tolist()

            logging.info(f"Found {len(papers_to_process_df)} papers to process.")
            logging.info(f"  - {len(new_papers)} new papers")
            logging.info(f"  - {len(changed_title_abstract)} papers with changed title/abstract (will re-embed)")
            logging.info(f"  - {len(unchanged_title_abstract)} papers with unchanged title/abstract (will copy embeddings)")

            # Drop the columns from the DB merge
            papers_to_process_df.drop(columns=[c for c in papers_to_process_df.columns if c.endswith("_db")], inplace=True)
        else:
            logging.info("No existing papers found for the given IDs. Processing all as new.")
            papers_to_process_df = df
    else:
        logging.info(f"Table '{TABLE_NAME}' not found. Processing all papers as new.")
        papers_to_process_df = df

    if papers_to_process_df.empty:
        logging.info("No new or changed papers to process. Exiting.")
        return

    df = papers_to_process_df

    voyageai_client = voyageai.Client(api_key=api_key)
    openai_client = OpenAI(api_key=openai_api_key)

    # --- Embedding ---
    # Only embed titles and abstracts that need re-embedding
    if "needs_embedding" in df.columns:
        papers_needing_embedding = df[df["needs_embedding"]]
        titles_to_embed = [t for t in papers_needing_embedding["title"].dropna().unique() if t]
        abstracts_to_embed = [a for a in papers_needing_embedding["abstract"].fillna("").unique() if a]
    else:
        # If no needs_embedding column, embed everything (for new tables)
        titles_to_embed = [t for t in df["title"].dropna().unique() if t]
        abstracts_to_embed = [a for a in df["abstract"].fillna("").unique() if a]

    logging.info(f"Embedding {len(titles_to_embed)} unique titles and {len(abstracts_to_embed)} unique abstracts...")
    title_embeddings = get_embeddings_in_batches(voyageai_client, openai_client, titles_to_embed)
    abstract_embeddings = get_embeddings_in_batches(voyageai_client, openai_client, abstracts_to_embed)
    logging.info("Embedding generation complete.")

    # --- Data Preparation ---
    final_df = pd.DataFrame()

    # Handle nested 'ids' columns safely
    if "ids" in df.columns:
        df["ids"] = df["ids"].apply(lambda x: {k: str(v) for k, v in x.items()} if isinstance(x, dict) else {})
    else:
        df["ids"] = pd.Series([{} for _ in range(len(df))])
    ids_df = pd.json_normalize(df["ids"])
    for col_name, source_key in [("mag_id", "mag"), ("pmid", "pmid"), ("pmcid", "pmcid"), ("arxiv_id", "arxiv")]:
        df[col_name] = ids_df[source_key] if source_key in ids_df.columns else ""

    string_cols = ["openalex_id", "doi", "publication_date", "title", "abstract", "mag_id", "pmid", "pmcid", "arxiv_id", "cited_by_api_url"]
    for col in string_cols:
        final_df[col] = df[col].fillna("").astype(str) if col in df.columns else ""

    authors_series = df["authors"].fillna(pd.Series([[] for _ in range(len(df))]))
    final_df["authors_id"] = authors_series.apply(lambda alist: [str(a.get("id", "")) for a in alist if isinstance(a, dict)])
    final_df["authors_name"] = authors_series.apply(lambda alist: [str(a.get("display_name", "")) for a in alist if isinstance(a, dict)])
    final_df["orcid"] = authors_series.apply(lambda alist: [str(a.get("orcid", "")) for a in alist if isinstance(a, dict)])

    topics_series = df["topics"].fillna(pd.Series([[] for _ in range(len(df))]))
    final_df["topics"] = topics_series.apply(lambda tlist: [str(t.get("display_name", "")) for t in tlist if isinstance(t, dict)])

    for col in ["raw_affiliation_strings", "institution_ids", "referenced_works", "citations"]:
        if col not in df.columns:
            final_df[col] = pd.Series([[] for _ in range(len(df))])
        else:
            final_df[col] = df[col].fillna(pd.Series([[] for _ in range(len(df))]))

    # Map embeddings only for papers that need them (don't overwrite copied embeddings)
    if "vector_title_voyageai" not in final_df.columns:
        final_df["vector_title_voyageai"] = final_df["title"].map(title_embeddings["voyageai"])
    else:
        # Only map for rows that don't already have embeddings
        mask_needs_mapping = final_df["vector_title_voyageai"].isna()
        final_df.loc[mask_needs_mapping, "vector_title_voyageai"] = final_df.loc[mask_needs_mapping, "title"].map(title_embeddings["voyageai"])

    if "vector_abstract_voyageai" not in final_df.columns:
        final_df["vector_abstract_voyageai"] = final_df["abstract"].map(abstract_embeddings["voyageai"])
    else:
        mask_needs_mapping = final_df["vector_abstract_voyageai"].isna()
        final_df.loc[mask_needs_mapping, "vector_abstract_voyageai"] = final_df.loc[mask_needs_mapping, "abstract"].map(abstract_embeddings["voyageai"])

    if "vector_title_openai" not in final_df.columns:
        final_df["vector_title_openai"] = final_df["title"].map(title_embeddings["openai"])
    else:
        mask_needs_mapping = final_df["vector_title_openai"].isna()
        final_df.loc[mask_needs_mapping, "vector_title_openai"] = final_df.loc[mask_needs_mapping, "title"].map(title_embeddings["openai"])

    if "vector_abstract_openai" not in final_df.columns:
        final_df["vector_abstract_openai"] = final_df["abstract"].map(abstract_embeddings["openai"])
    else:
        mask_needs_mapping = final_df["vector_abstract_openai"].isna()
        final_df.loc[mask_needs_mapping, "vector_abstract_openai"] = final_df.loc[mask_needs_mapping, "abstract"].map(abstract_embeddings["openai"])

    # Fill missing embeddings with zeros
    for col in ["vector_title_voyageai", "vector_abstract_voyageai", "vector_title_openai", "vector_abstract_openai"]:
        final_df[col] = final_df[col].apply(lambda x: x if isinstance(x, (list, np.ndarray)) else np.zeros(EMBEDDING_DIM))

    # --- Update DB ---
    if not final_df.empty:
        try:
            if table is None:
                logging.info(f"Creating new table '{TABLE_NAME}' and adding {len(final_df)} papers.")
                db.create_table(TABLE_NAME, data=final_df, schema=Paper)
            else:
                if ids_to_delete:
                    logging.info(f"Deleting {len(ids_to_delete)} old versions of papers...")
                    delete_chunk_size = 500
                    for i in range(0, len(ids_to_delete), delete_chunk_size):
                        id_chunk = ids_to_delete[i : i + delete_chunk_size]
                        id_filter_str = ", ".join([f"'{_id}'" for _id in id_chunk])
                        if id_filter_str:
                            table.delete(f"openalex_id IN ({id_filter_str})")
                    logging.info("Deletion of old versions complete.")

                logging.info(f"Adding {len(final_df)} new/updated papers to table '{TABLE_NAME}'.")
                table.add(data=final_df)

            logging.info(f"Successfully processed {len(final_df)} papers.")

        except Exception as e:
            logging.error(f"An error occurred during DB update for {input_path}: {e}")

    else:
        logging.info("Final dataframe is empty, nothing to add to the database.")

    del df, final_df, title_embeddings, abstract_embeddings


if __name__ == "__main__":
    load_dotenv()
    input_file = "/home/ubuntu/openalex/workspace/zimo.li/task5v3files/2023-11-12.jsonl"

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            filename="/home/ubuntu/openalex/workspace/zimo.li/build_db.log",
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )

    voyage_api_key = os.environ.get("VOYAGE_API_KEY")
    if not voyage_api_key:
        raise ValueError("VOYAGE_API_KEY environment variable not set")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    failed_files = []
    try:
        update_papers(input_file, voyage_api_key, openai_api_key)
    except Exception as e:
        logging.error(f"Error processing file {input_file}: {e}")
        failed_files.append(input_file)

    # Retry logic for failed files
    if failed_files:
        logging.info("\nRetrying failed files...")
        for file in failed_files:
            try:
                update_papers(file, voyage_api_key, openai_api_key)
                logging.info(f"Successfully retried file: {file}")
            except Exception as e:
                logging.warning(f"Retry failed for file {file}: {e}")
