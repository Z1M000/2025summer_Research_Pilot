'''
Task10 Crossref Notes:
1.	Logic: For each page of each type (journal + proceedings), build paper list → merge OpenAlex info → upsert into LanceDB → save checkpoint.
2.  Andrew's task 7 doesn't have voyageai, so I don't have this either. I do have openai's embedding
3.	Run setup:
	•	Create a checkpoint file (same format as mine).
	•	Manually update from_update_date. If it's a new date, reset both start_cursor to "*".
	•	One-time: create scalar index on doi.
	•	Change page limit, probably raise ROWS (max 2000).
4.	To Yuntong: Plz test and let me know improvements. Plzz don’t try it on the papers.lance table yet.
'''
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

# ---------------------- Config ----------------------
DB_PATH = "/home/ubuntu/database"
TABLE_NAME = "task10crossref"
OPENAI_MODEL_ID = "text-embedding-3-small"
EMBEDDING_DIM = 1024
ROWS = 10
TYPES = ["journal-article", "proceedings-article"]
CHECKPOINT_PATH = "openalex/workspace/zimo.li/task10_crossref/checkpoint.json"

# ---------------------- load Andrew's embedding method in task 7----------------------
sys.path.append("/home/ubuntu/openalex/workspace/andrew.jaffe")
import task7 as andrew_task7
andrew_get_embeddings = andrew_task7.get_embeddings_in_batches

# ---------------------- Create OpenAI Client ----------------------
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
openai_client = OpenAI(api_key=openai_api_key)

# ---------------------- Schema ----------------------
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

# ---------------------- Helpers Start----------------------
def iter_crossref_pages(
    from_update_date: str,
    paper_type: str,
    rows: int,
    start_cursor: str | None = None,
    page_limit: int | None = None,
):
    base = "https://api.crossref.org/works"
    select_fields = ["DOI","title","abstract","author","created","reference"]

    def build_params(t: str, cursor: str):
        filters = f"type:{paper_type},from-update-date:{from_update_date}"
        params = {
            "filter": filters,
            "rows": str(rows),
            "cursor": cursor,
            "select": ",".join(select_fields),
        }
        return params

    pages_emitted = 0

    cursor = start_cursor or "*"
    while True:
        params = build_params(paper_type, cursor)
        try:
            r = requests.get(base, params=params, timeout=30)
            r.raise_for_status()
            msg = r.json().get("message", {}) or {}
            items = msg.get("items", []) or []
            next_cursor = msg.get("next-cursor")
        except Exception as e:
            print(f"[Crossref] request failed: {e}")
            return

        yield items, next_cursor, {"type": paper_type, "cursor": cursor}

        # when reach page limit end
        pages_emitted += 1
        if not next_cursor or not items:
            break
        if page_limit and pages_emitted >= page_limit:
            return

        cursor = next_cursor
        time.sleep(0.2)  # avoid 429

def zero_vec():
    return [0.0] * EMBEDDING_DIM

def clean_abstract(raw: str | None, keep_headers: bool = False) -> str:
    if not raw or not str(raw).strip():
        return ""
    soup = BeautifulSoup(raw, "html.parser")

    # 1) remove "Abstract/ABSTRACT/Summary"
    for t in soup.find_all(["title", "jats:title"]):
        if t.get_text(strip=True).lower() in {"abstract", "summary"}:
            t.decompose()

    # 2) remove italic/bold
    for t in soup.find_all(["i", "em", "jats:italic", "b", "strong", "jats:bold"]):
        t.replace_with(t.get_text())

    # 3) subtitle
    parts = []
    secs = soup.find_all(["jats:sec", "sec"])
    if secs:
        for sec in secs:
            header = ""
            if keep_headers:
                tt = sec.find(["jats:title", "title"])
                if tt:
                    header = tt.get_text(" ", strip=True) + ": "
                    tt.decompose()
            body = " ".join(p.get_text(" ", strip=True) for p in sec.find_all(["jats:p", "p"]))
            if body:
                parts.append(header + body)
    else:
        # 没有 <sec> 就直接拿所有段落；没有段落就拿全文
        ps = soup.find_all(["jats:p", "p"])
        text = " ".join(p.get_text(" ", strip=True) for p in ps) if ps else soup.get_text(" ", strip=True)
        parts = [text]

    text = "\n".join(p for p in parts if p)

    # 4) Unicode 归一化 + 空白清理
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00A0", " ").replace("\u2009", " ").replace("\u200A", " ").replace("\u200B", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text

def get_authors_name_only(item: dict) -> List[str]:
    authors = item.get("author", []) or []
    return [f"{(a.get('given') or '').strip()} {(a.get('family') or '').strip()}".strip()
            for a in authors if isinstance(a, dict)]

def get_refs(item: dict) -> List[str]:
    out = []
    for ref in item.get("reference", []) or []:
        doi = (ref or {}).get("DOI", "")
        if doi:
            out.append(f"https://doi.org/{doi}")
    return out

def to_dict(p: Paper) -> Dict[str, Any]:
    try:
        return p.model_dump()
    except Exception:
        return p.dict()

def fetch_existing_by_dois(table, dois: List[str]) -> Dict[str, dict]:
    if not dois:
        return {}

    # IN 列表别太长，分片稳妥
    CHUNK = 500
    result: Dict[str, dict] = {}

    def _q(s: str) -> str:
        return "'" + (s or "").replace("'", "''") + "'"

    for i in range(0, len(dois), CHUNK):
        chunk = dois[i:i + CHUNK]
        where = f"doi IN ({','.join(_q(d) for d in chunk)})"

        # 新 API：从空搜索器开始，只用 where 做标量筛选
        rows = (
            table.search()          # 空搜索（不做向量检索）
                 .where(where)      # 纯标量过滤
                 .limit(len(chunk) + 50)
                 .to_list()
        )
        for r in rows or []:
            d = r.get("doi")
            if d:
                result[d] = r

    return result

# ---------------------- Helpers End----------------------

# ---------------------- Main ----------------------
if __name__ == "__main__":
    # connect db
    db = lancedb.connect(DB_PATH)
    table = db.open_table(TABLE_NAME)
    table.create_scalar_index("doi", replace=True)

    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    update_date = state.get("from_update_date")
    

    # Fetch page by page (after finishing each page, immediately upsert + save checkpoint)
    for paper_type in TYPES:
        start_cursor = state.get(paper_type).get("next_cursor", "*")
        print(f"loaded checkpoint: cursor={start_cursor}")
        for items, next_cursor, meta in iter_crossref_pages(
            from_update_date= update_date,
            paper_type = paper_type,
            rows=ROWS,
            start_cursor=start_cursor,
            page_limit=2
        ):
            print("\n--page starts--")
            if not items:
                print("No data from Crossref. Exit.")
                break

            incoming_dois = [f"https://doi.org/{it.get('DOI','')}".strip() for it in items if it.get("DOI")]
            existing_map = fetch_existing_by_dois(table, incoming_dois)

            incoming = []
            titles_need_embed, abstracts_need_embed = [], []

            for it in items:
                doi_full = f"https://doi.org/{it.get('DOI','')}".strip()

                created = (it.get("created", {}) or {}).get("date-parts", [[]])
                date_parts = created[0] if created and isinstance(created, list) else []
                pub_date = "-".join(f"{p:02d}" for p in date_parts) if date_parts else ""

                title_list = it.get("title") or []
                new_title = (title_list[0] if title_list else "").strip()
                new_abs = clean_abstract(it.get("abstract")).strip()
                old = existing_map.get(doi_full)

                if old:
                    p = Paper(
                        openalex_id = old.get("openalex_id",""),
                        doi=doi_full, 
                        publication_date=pub_date,
                        title=new_title, 
                        abstract=new_abs,
                        vector_title_voyageai=zero_vec(), 
                        vector_abstract_voyageai=zero_vec(),
                        vector_title_openai=zero_vec(), 
                        vector_abstract_openai=zero_vec(),
                        mag_id=old.get("mag_id"), 
                        pmid= old.get("pmid"), 
                        pmcid= old.get("pmcid"), 
                        arxiv_id= old.get("arxiv_id"),
                        raw_affiliation_strings=old.get("raw_affiliation_strings"), 
                        institution_ids=old.get("institution_ids"), 
                        authors_id= old.get("authors_id"),
                        authors_name=get_authors_name_only(it), 
                        orcid=old.get("orcid"),
                        referenced_works=get_refs(it), 
                        topics=old.get("topics"), 
                        cited_by_api_url=old.get("cited_by_api_url"), 
                        citations=old.get("citations")
                    )
                    if (p.abstract or "").strip() == "":
                        p.abstract = (old.get("abstract") or "")
                        p.vector_abstract_openai = old.get("vector_abstract_openai") or zero_vec()
                    else:
                        if (old.get("abstract") or "").strip() == p.abstract:
                            p.vector_abstract_openai = old.get("vector_abstract_openai") or zero_vec()
                    old_title = (old.get("title") or "").strip()
                    if old_title and new_title and old_title.lower() == new_title.lower():
                        p.vector_title_openai = old.get("vector_title_openai") or zero_vec()
                else:
                    p = Paper(
                        openalex_id="", 
                        doi=doi_full, 
                        publication_date=pub_date,
                        title=new_title, 
                        abstract=new_abs,
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
                        authors_name=get_authors_name_only(it), 
                        orcid=[],
                        referenced_works=get_refs(it), 
                        topics=[], 
                        cited_by_api_url="", 
                        citations=[]
                    )

                if (p.title or "").strip() and p.vector_title_openai == zero_vec():
                    titles_need_embed.append(p.title)
                if (p.abstract or "").strip() and p.vector_abstract_openai == zero_vec():
                    abstracts_need_embed.append(p.abstract)

                incoming.append(p)
                print(f"appended paper: {doi_full}")

            # embeddings
            titles_need_embed = list(set(titles_need_embed))
            abstracts_need_embed = list(set(abstracts_need_embed))

            title2vec, abs2vec = {}, {}
            if titles_need_embed:
                out = andrew_get_embeddings(voyageai_client=None, openai_client=openai_client, texts=titles_need_embed)
                title2vec = out.get("openai", {}) if isinstance(out, dict) else {}
                print("andrew ai p1 done"); print("andrew ai p2 done")

            if abstracts_need_embed:
                out = andrew_get_embeddings(voyageai_client=None, openai_client=openai_client, texts=abstracts_need_embed)
                abs2vec = out.get("openai", {}) if isinstance(out, dict) else {}

            for p in incoming:
                if p.title and p.vector_title_openai == zero_vec():
                    v = title2vec.get(p.title)
                    if v: p.vector_title_openai = v
                if p.abstract and p.vector_abstract_openai == zero_vec():
                    v = abs2vec.get(p.abstract)
                    if v: p.vector_abstract_openai = v

            records = [to_dict(p) for p in incoming]
            (
                table.merge_insert("doi")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(records)
            )

            table.optimize()  # scalar index update
            print(f"merge_insert for current page done. total={len(records)}, existing={len(existing_map)}")

            # save checkpoint
            state[paper_type] = {
                "next_cursor": next_cursor,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            print("-- page ends --\n")

        print(f"DONE for one type \n ************** \n")
    
    print("ALL DONE!!")

