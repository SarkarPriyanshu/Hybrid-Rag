from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from app.utils.data_preprocess import build_text_data
from app.models.embeddings_miniLM_l6_v2 import MiniLMEmbeddings
from app.models.language_model import query_google_llm
from app.utils.extract_data_from_source import extract_spl_info
from app.utils.vector_db import PineconeVectorDB
from app.utils.setup_db import PostgresDB
from app.utils.custom_logging import logger
from config import config
import httpx
import xmltodict
import pandas as pd
import os
import asyncio

# --------------------------
# FastAPI app
# --------------------------
app = FastAPI(title="DailyMed SPL CSV Generator")
logger.info("[INIT] FastAPI app created: DailyMed SPL CSV Generator")

# --------------------------
# Config
# --------------------------
general = config.components['general']
db_conf = config.components['db']
vector_db_api_key = config.components['vectordb'].vector_db_api_key

DATA_FOLDER = general.data_folder
CSV_PATH = general.csv_path
os.makedirs(DATA_FOLDER, exist_ok=True)

BASE_URL = general.base_url
MAX_ROWS = general.max_rows
PAGE_SIZE = general.page_size
PAGE_NO = general.page_no
DB_TABLE_NAME = db_conf.db_table_name

logger.info(f"[CONFIG] DATA_FOLDER={DATA_FOLDER}")
logger.info(f"[CONFIG] CSV_PATH={CSV_PATH}")
logger.info(f"[CONFIG] BASE_URL={BASE_URL}")
logger.info(f"[CONFIG] MAX_ROWS={MAX_ROWS}, PAGE_SIZE={PAGE_SIZE}, PAGE_NO={PAGE_NO}")
logger.info(f"[CONFIG] DB_TABLE_NAME={DB_TABLE_NAME}")

# Global objects
postgres_db: PostgresDB = None
embedding_model: MiniLMEmbeddings = None
pinecone_db: PineconeVectorDB = None  # <-- make Pinecone global

# Startup: DB & Embedding model
@app.on_event("startup")
async def startup_event():
    global postgres_db, embedding_model
    logger.info("[STARTUP] Initializing PostgreSQL client...")
    postgres_db = PostgresDB(db_conf)
    logger.info("[STARTUP] PostgreSQL client ready.")

    logger.info("[STARTUP] Loading MiniLM-L6-v2 embedding model...")
    embedding_model = MiniLMEmbeddings()
    logger.info("[STARTUP] Embedding model loaded.")

# Async utility: fetch SPL JSON
async def fetch_spl_page(client: httpx.AsyncClient, page_no: int):
    url = f"{BASE_URL}/spls.json?pagesize={PAGE_SIZE}&page={page_no}"
    resp = await client.get(url)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return [spl.get("setid") for spl in data if "setid" in spl]

# Async utility: fetch SPL XML
async def fetch_spl_xml(client: httpx.AsyncClient, setid: str):
    url = f"{BASE_URL}/spls/{setid}.xml"
    resp = await client.get(url)
    resp.raise_for_status()
    data_dict = xmltodict.parse(resp.content)
    return extract_spl_info(data_dict)

# Gather Data Pipeline
@app.get("/gather_data_pipeline")
async def generate_csv():
    global pinecone_db  # <-- use global pinecone_db
    logger.info("[API] /gather_data_pipeline called")

    all_setids = []
    async with httpx.AsyncClient(timeout=60) as client:
        tasks = [fetch_spl_page(client, page) for page in range(1, PAGE_NO + 1)]
        pages_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in pages_results:
            if isinstance(result, Exception):
                logger.error(f"[FETCH ERROR] {result}")
                continue
            all_setids.extend(result)
            if MAX_ROWS and len(all_setids) >= MAX_ROWS:
                all_setids = all_setids[:MAX_ROWS]
                break

        if not all_setids:
            logger.warning("[NO DATA] No SPL setIds found")
            return {"message": "No SPL setIds found; CSV not generated."}

        logger.info(f"[PROCESS] Total SPL setIds to process: {len(all_setids)}")

        sem = asyncio.Semaphore(8)
        async def safe_fetch(setid):
            async with sem:
                try:
                    return await fetch_spl_xml(client, setid)
                except Exception as e:
                    logger.warning(f"[PARSE FAIL] setid={setid} exception={str(e)}")
                    return None

        xml_tasks = [safe_fetch(setid) for setid in all_setids]
        rows = [res for res in await asyncio.gather(*xml_tasks) if res is not None]

    if not rows:
        logger.warning("[NO ROWS] No SPL data extracted")
        return {"message": "No SPL data extracted; CSV not generated."}

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    logger.info(f"[CSV] Saved SPL data CSV -> {CSV_PATH} | rows={len(df)}")

    # Prepare DB + RAG Data
    db_data, text_data = build_text_data(df)

    # Upsert to Pinecone
    if pinecone_db is None:
        pinecone_db = PineconeVectorDB(
            api_key=vector_db_api_key,
            index_name="all-minilm-l6-v2-index",
            dimension=384,
            embedding_model=embedding_model
        )

    batch_size = 16
    for i in range(0, len(text_data), batch_size):
        pinecone_db.upsert_documents(text_data[i:i+batch_size])

    # Insert into PostgreSQL
    postgres_db.insert_dataframe(db_data, table_name=DB_TABLE_NAME)
    logger.info(f"[DB INSERT] Table={DB_TABLE_NAME} | rows={len(db_data)}")

    return FileResponse(
        CSV_PATH,
        media_type="text/csv",
        filename=os.path.basename(CSV_PATH)
    )

# Ask Endpoint
@app.get("/ask")
async def ask_user(query: str = Query(..., description="User query for RAG")):
    """
    Endpoint to receive a user query and return an answer.
    Uses PostgresDB.query_with_llm to convert natural language to SQL and fetch results.
    Returns structured JSON with column names and row values.
    """
    global pinecone_db
    if not postgres_db:
        return {
            "query": query,
            "db_answer": [],
            "vector_answer": [],
            "llm_response": "PostgreSQL client not initialized yet.",
            "sources": []
        }

    try:
        # Run NL -> SQL -> Postgres
        raw_result = postgres_db.query_with_llm(query)

        # Query vector DB (Pinecone)
        vector_context = []
        if pinecone_db:
            vector_context = pinecone_db.query(query, k=5)

        # LLM combines context
        llm_response = query_google_llm(
            db_answer=raw_result,
            vector_context=vector_context,
            user_query=query
        )

        return {
            "query": query,
            "db_answer": raw_result,
            "vector_answer": vector_context,
            "llm_response": llm_response,
            "sources": ["postgres", "vector"]
        }

    except Exception as e:
        logger.error(f"[ASK ERROR] Failed to run query_with_llm: {str(e)}")
        return {
            "query": query,
            "db_answer": [],
            "vector_answer": [],
            "llm_response": f"Error processing your query: {str(e)}",
            "sources": []
        }
