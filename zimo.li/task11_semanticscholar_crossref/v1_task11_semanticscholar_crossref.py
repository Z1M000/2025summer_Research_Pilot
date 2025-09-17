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
import pandas as pd


# ---------------config----------------
DB_PATH = "/home/ubuntu/database"
TABLE_NAME = "task11test_zimo"
EMBEDDING_DIM = 1024
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

def get_refs(item):
    refs = item.get("references", [])
    if refs is None:
        # print("No references found")
        return []
    
    ref_list = []
    for ref in refs:
        paperId = ref.get("paperId", "")
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paperId}?fields=externalIds"
        try:
            r = requests.get(url)
            if r.status_code != 200:
                print(f"Failed to fetch data for reference {ref}, status code: {r.status_code}")
                continue
            ref_item = r.json()
            doi_full = ref_item.get("externalIds", {}).get("DOI", "")
            if doi_full:
                doi_full = f"https://doi.org/{doi_full}"
                ref_list.append(doi_full)
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching data for reference {ref}: {e}")
            continue
    return ref_list
    
def get_paper_by_paperId(paperId):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paperId}?fields=title,abstract,externalIds,references"
    print("getting paper metadata at url:", url)
    try:
        r = requests.get(url)
        if r.status_code != 200:
            print(f"Failed to fetch data for paperId {paperId}, status code: {r.status_code}")
            return None
        item = r.json()
        doi_full = item.get("externalIds", {}).get("DOI")
        if not doi_full:
            doi_full = ""
        else:
            doi_full = f"https://doi.org/{doi_full}"
        title=(item.get("title") or "").strip()
        abstract=(item.get("abstract") or "").strip()
        ref_list = get_refs(item)
        
        print("creating Paper object...")
        print("doi_full:", doi_full)
        print("title:", title[:80])
        print("abstract:", abstract[:80])
        print("references:", ref_list[:2])  
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
    except Exception as e:
        print(f"Error fetching data for paperId {paperId}: {e}")
        return None

    return p
# ---------------helpers end----------------

# def get_cite_crossref(doi: str) -> List[Paper]: # CrossRef simply doesn't provide citation data
#     citations = []
#     doi = doi.replace("https://doi.org/", "")
#     url = f"https://api.crossref.org/works/{doi}"
#     print(url)
#     return citations

def get_cite_semanticscholar(doi: str) -> List[Paper]:
    
    citation_papers = []
    doi = doi.replace("https://doi.org/", "")
    
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}/citations"
    print(f"getting citations list for DOI: {doi}")
    print(f"url: {url}")
    # we need a list of Paper objects with title, abstract, DOI, reference list (optional)
    # https://api.semanticscholar.org/graph/v1/paper/DOI:10.1002/cpp.646?fields=title,abstract,externalIds,references

    try:
        r = requests.get(url)
        if r.status_code != 200:
            print(f"Failed to fetch citations for DOI {doi}, status code: {r.status_code}")
            return citation_papers
        cite_list = r.json().get("data", [])
        print(f"found {len(cite_list)} citations")
        # print(cite_list)
        print("got the cite_list, starting creating Paper objects...")
        for cite in cite_list[4:20]:
            paperId = cite.get("citingPaper", {}).get("paperId", "")
            print(f"\nprocessing paperId: {paperId}")
            if paperId:
                p = get_paper_by_paperId(paperId)
                if p:
                    citation_papers.append(p)
                time.sleep(1)

    except Exception as e:
        print(f"Error fetching data for DOI {doi}: {e}")

    return citation_papers


# ---------------main----------------
if __name__ == "__main__":
    # db = lancedb.connect(DB_PATH)
    # table = db.open_table(TABLE_NAME)
    # table.create_scalar_index("doi")
    get_cite_semanticscholar(the_doi)