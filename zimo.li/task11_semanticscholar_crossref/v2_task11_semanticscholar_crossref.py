# Paper id = None? now I just skip. shall I try title?
# 翻页，回答yuntong abstract缺的多不多
# retry logic for 429

import os
import sys
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
import re
import unicodedata
import json
import lancedb
from lancedb.pydantic import LanceModel, Vector
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import time
import random
import pandas as pd


# ---------------config----------------
DB_PATH = "/home/ubuntu/database"
TABLE_NAME = "task11test_zimo"
EMBEDDING_DIM = 1 # should be 1024
BATCH_SIZE_PAPER_ID = 100
BATCH_SIZE_REF = 200
LIMIT = 100 # per page
MAX_ATTEMPTS = 5
the_doi = "https://doi.org/10.1002/cpp.646"
dois = ['https://doi.org/10.1067/mva.1987.avs0060535',
 'https://doi.org/10.1002/cpp.646',
 'https://doi.org/10.1067/mva.1988.avs0070318',
 'https://doi.org/10.1002/ijc.2910340402',
 'https://doi.org/10.1002/anie.201204963']

# schema
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



# ---------------helpers start----------------
def zero_vec():
    return [0.0] * EMBEDDING_DIM

def get_all_citation_ids(doi: str) -> list[str]:
    doi = doi.replace("https://doi.org/", "")
    url = f"https://api.semanticscholar.org/graph/v1/paper/{doi}/citations"
    print("getting citation ids from", url)
    all_ids = []
    offset = 0
    attempt = 0

    while True:
        r = requests.get(url, params={"fields": "citingPaper.paperId",
                                      "limit": LIMIT, "offset": offset})
        if r.status_code == 429:
            while attempt < MAX_ATTEMPTS:
                wait = wait_time_429(r, attempt=attempt)
                # print("** encountered 429, retrying after", wait, "seconds")
                time.sleep(wait)
                r = requests.get(url, params={"fields": "citingPaper.paperId",
                                      "limit": LIMIT, "offset": offset})
                if r.status_code != 429:
                    break
                attempt += 1
            if attempt == MAX_ATTEMPTS:
                print("max attempts reached for 429, skip this request to get citation ids for doi", doi)
                break
        elif r.status_code != 200:
            print(f"Error getting citation ids: {r.status_code}")
            break
        data = r.json().get("data", [])
        if not data:
            break
        all_ids.extend([c["citingPaper"]["paperId"] for c in data if "citingPaper" in c])
        offset += LIMIT
    
    return all_ids

def create_paper(item):
    paperId = item.get("paperId", "")
    # print(f"\nprocessing paperId: {paperId}")
    doi_full = item.get("externalIds", {}).get("DOI")
    if not doi_full:
        doi_full = ""
    else:
        doi_full = f"https://doi.org/{doi_full}"
    title=(item.get("title") or "").strip()
    abstract=(item.get("abstract") or "").strip()
    ref_list = batch_get_refs(item)
     
    p = Paper(
        openalex_id="", 
        doi=doi_full,
        publication_date="",
        title=title,
        abstract=abstract,
        vector_title_voyageai=zero_vec(), 
        vector_abstract_voyageai=zero_vec(),
        vector_title_openai=zero_vec(), 
        vector_abstract_openai=zero_vec(),
        mag_id="", 
        pmid="", 
        pmcid="", 
        arxiv_id="",
        raw_affiliation_strings=[], 
        institution_ids=[], 
        authors_id=[],
        authors_name=[], 
        orcid=[],
        referenced_works=ref_list, 
        topics=[], 
        cited_by_api_url="", 
        citations=[]
    )

    # print("created Paper object for paperId:", paperId[:5])
    # print("    doi_full:", doi_full)
    # print("    title:", title[:80])
    # print("    abstract:", abstract[:80])
    # print("    references:", ref_list[:2]) 
    print("created Paper for", paperId[:5], "with abstract length", len(abstract), "and", len(ref_list), "references")
    return p

def batch_get_refs(item):
    refs = item.get("references", [])
    if not refs:
        return []
    
    # get paperIds in refs and map to DOIs
    paperIds = [ref.get("paperId") for ref in refs if ref.get("paperId")]
    if not paperIds:
        return []
    
    ref_list = []
    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    params = {"fields": "externalIds"}

    for i in range(0, len(paperIds), BATCH_SIZE_REF):
        batch_ids = paperIds[i:i+BATCH_SIZE_REF]
        attempt = 0
        try:
            r = requests.post(url, params=params, json={"ids": batch_ids}, timeout=30)
            if r.status_code == 429:
                while attempt < MAX_ATTEMPTS:
                    wait = wait_time_429(r, attempt=attempt)
                    # print("** encountered 429, retrying after", wait, "seconds")
                    time.sleep(wait)
                    r = requests.post(url, params=params, json={"ids": batch_ids}, timeout=30)
                    if r.status_code != 429:
                        break
                    attempt += 1
                if attempt == MAX_ATTEMPTS:
                    print("max attempts reached for 429, skip this request to get citation ids for doi", doi)
                    break
            elif r.status_code != 200:
                print(f"Failed to fetch batch {i//BATCH_SIZE+1}, status: {r.status_code}")
                continue
            
            # with open("openalex/workspace/zimo.li/task11_semanticscholar_crossref/batch_test.json", "w") as f:
            #     json.dump(r.json(), f, indent=2)
            data = r.json()
            for ref_item in data:
                doi_full = ref_item.get("externalIds", {}).get("DOI", "")
                if doi_full:
                    doi_full = f"https://doi.org/{doi_full}"
                    ref_list.append(doi_full)

        except Exception as e:
            print(f"Error fetching batch {i//BATCH_SIZE+1}: {e}")
            continue
    
    return ref_list
   
def batch_get_paper_by_paperId(paperIds):
    papers = []
    attempt = 0
    r = requests.post(
        'https://api.semanticscholar.org/graph/v1/paper/batch',
        params={'fields': 'title,abstract,externalIds,references'},
        json={"ids": paperIds}
    )
    if r.status_code == 429:
        while attempt < MAX_ATTEMPTS:
            wait = wait_time_429(r, attempt=attempt)
            # print("** encountered 429, retrying after", wait, "seconds")
            time.sleep(wait)
            r = requests.post(
                'https://api.semanticscholar.org/graph/v1/paper/batch',
                params={'fields': 'title,abstract,externalIds,references'},
                json={"ids": paperIds}
            )
            if r.status_code != 429:
                break
            attempt += 1
            if attempt == MAX_ATTEMPTS:
                print("max attempts reached for 429, skip this request to get citation ids for doi", doi)
                break

    elif r.status_code != 200:
        print("Error:", r.status_code, r.text)
        return papers
    # with open("openalex/workspace/zimo.li/task11_semanticscholar_crossref/batch_test.json", "w") as f:
    #     json.dump(r.json(), f, indent=2)
    items = r.json()
    # print(f"got {len(items)} items in batch")
    for item in items:
        p = create_paper(item)
        papers.append(p)
    return papers

def wait_time_429(resp, attempt, base=1, cap=30):
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    # 没有 Retry-After 时，指数退避 + 抖动
    wait = min(cap, base * (2 ** attempt))
    jitter = random.uniform(0, wait / 2)
    return wait + jitter   

# ---------------helpers end----------------

# def get_cite_crossref(doi: str) -> List[Paper]: # CrossRef simply doesn't provide citation data
#     citations = []
#     doi = doi.replace("https://doi.org/", "")
#     url = f"https://api.crossref.org/works/{doi}"
#     print(url)
#     return citations

def get_cite_semanticscholar(doi: str) -> List[Paper]:    
    citation_papers = []
    # we need a list of Paper objects with title, abstract, DOI, reference list (optional)
    try:
        paperIds = get_all_citation_ids(doi)
        print("total citation paperIds:", len(paperIds))
        for i in range(0, len(paperIds), BATCH_SIZE_PAPER_ID):
            batch_ids = paperIds[i:i+BATCH_SIZE_PAPER_ID]
            batch_papers = batch_get_paper_by_paperId(batch_ids)
            citation_papers.extend(batch_papers)
    
    except Exception as e:
        print(f"Error fetching data for DOI {doi}: {e}")

    return citation_papers


# ---------------main----------------
if __name__ == "__main__":
    # db = lancedb.connect(DB_PATH)
    # table = db.open_table(TABLE_NAME)
    # table.create_scalar_index("doi")
    
    # for doi in dois:
    #     print(f"\nProcessing DOI: {doi}")
    #     papers = get_cite_semanticscholar(doi)
    #     print(f"Retrieved {len(papers)} citation papers for DOI {doi}")

    get_cite_semanticscholar("10.18653/v1/N18-3011")
    # get_cite_semanticscholar(the_doi)
    
    # paperIds = [
    #     "a2c79802bbdea1781bde8a6c62e294682f6da987", "b54517f441f6122c288ef0b21626ae0eb43eb0aa", 
    #     "8b729e47d2f0f3ef5a9821a4580fe36d2b8ca0d7", "3274e85c8abb41e2c81155f5d01bb03a405f9a3e", 
    #     "ca712d9e8167cd30209fcbbbb38d234cbeacb5dc", "173dcfe3c131d91cadd030c39c2cd9c31b0ac153",
    #     "d39e71df539cc09015262c31fa6e650d103aea70", "e637be82f448217ac3e066a95445f0c302d0c7f5"
    # ]
    # batch_get_paper_by_paperId(paperIds)