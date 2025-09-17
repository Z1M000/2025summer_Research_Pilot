# Task 5 
# Details:
# 1. id --> openalex_id, doi --> doi. I put them at the front for now.
#    feel free to change any order!
# 2. I included "openalex_id", "doi", "mag_id", "pmid",  "pmcid", "arxiv_id". pmcid and arxiv_id are lowkey rare.
#    I sent you(yuntong) a screenshot of the counter results, plzz decide whether to keep them
# 3. works2lancedb method **ADDS** data from the input file to the lancedb table.
#    **so the person in Task 7 needs to initialize the table first**
# 4. I used the official gemini embedding model. the default dimension is 3072, which can be changed
# 5. it's suggested to update the table in batches, avoiding creating too many versions of the db.

import json
from google import genai
import lancedb
import os
from dotenv import load_dotenv 
import pandas as pd



db = lancedb.connect("./lancedb")
table = db.open_table("test_openalex")
# table.delete()
input_path = "/home/ubuntu/openalex/workspace/zimo.li/test_input.jsonl"
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
model_id = "gemini-embedding-001" # current official geimini embedding model, defualt 3072 dimensions
table_name = "test_openalex"
# data = []


def works2lancedb(input_path):
    rows = []
    with open(input_path, "r") as f:
        for line in f:
            try:
                paper = json.loads(line)
                rows.append(convert(paper))
            except Exception as e:
                print(f"Error processing line: {line.strip()}. Error: {e}")
                break

    df = pd.DataFrame(rows)
    # if table_name in db.table_names():
    #     db.drop_table(table_name)
    # db.create_table(table_name, data=df)
    table.add(data = df)
    print(f"Table '{table_name}' updated with {len(df)} rows.")



def getEmbedding(text):
    if text is None or text == "":
        return None
    try:
        result = client.models.embed_content(
            model= model_id, 
            contents= text) # demension default to be 3072, change if wanted
        [embedding] = result.embeddings
        return embedding.values
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None



def convert(paper):
    ids = paper.get("ids", {})
    authors = paper.get("authors", [])
    return{
        "openalex_id": paper.get("id"),
        "doi": paper.get("doi"), # shall I put these two ids at the front like this or with other ids?
        "publication_date": paper.get("publication_date"),
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "title_embedding": getEmbedding(paper.get("title")),
        "abstract_embedding": getEmbedding(paper.get("abstract")),
        "mag_id": ids.get("mag"),
        "pmid": ids.get("pmid"),
        "pmcid": ids.get("pmcid"),
        "arxiv_id": ids.get("arxiv"),
        "raw_affiliation_strings": paper.get("raw_affiliation_strings", []),
        "institution_ids": paper.get("institution_ids", []),
        "authors_id": [a.get("id") for a in authors],
        "authors_name": [a.get("display_name") for a in authors],
        "orcid": [a.get("orcid") for a in authors],
        "referenced_works": paper.get("referenced_works", []),
        "topics": paper.get("topics", []),
        "cited_by_api_url": paper.get("cited_by_api_url"),
        "citations": [],
    }

works2lancedb(input_path)

# rows = [convert(paper) for paper in data]
# df = pd.DataFrame(rows)
# db.drop_table("test_openalex")
# table = db.create_table("test_openalex", data = df, exist_ok = True)
# table = db.open_table(table_name)
# df = table.to_pandas()
# print(df.head())

