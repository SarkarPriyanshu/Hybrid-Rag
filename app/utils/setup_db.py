# app/utils/setup_db.py

import time
import re
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from app.models.language_model import extract_main_entity
from sqlalchemy import inspect
from config import config
from app.utils.custom_logging import logger

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.sql.base import SQLDatabaseChain
from langchain_community.utilities.sql_database import SQLDatabase


class PostgresDB:
    def __init__(self, db_config=None, llm_model=None, api_key=None):
        self.db_config = db_config or config.components["db"]
        self.db_url = self.db_config.db_url
        self.engine = None
        self._connect()

        llm_model = llm_model or config.components["llm"].model_name
        api_key = api_key or config.components["llm"].api_key
        if not api_key:
            raise ValueError("[PostgresDB] Missing LLM API Key")

        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0, api_key=api_key)

        self.schema_context = self._get_db_schema()
        logger.info(f"[PostgresDB] Initial schema context: {self.schema_context}")

        self._init_sql_chain()

    # ----------------------
    def _connect(self):
        logger.info(f"[PostgresDB] Connecting to PostgreSQL at: {self.db_url}")
        self.engine = sa.create_engine(self.db_url, pool_pre_ping=True, future=True)
        self._wait_for_db()
        logger.info("[PostgresDB] Connected successfully.")

    def _wait_for_db(self, timeout=180, interval=2):
        logger.info("[PostgresDB] Checking PostgreSQL readiness...")
        start_time = time.time()
        while True:
            try:
                with self.engine.connect() as conn:
                    conn.execute(sa.text("SELECT 1"))
                logger.info("[PostgresDB] PostgreSQL is up and reachable.")
                return
            except OperationalError as e:
                if time.time() - start_time > timeout:
                    logger.error(f"[PostgresDB] Timeout: DB not ready: {e}")
                    raise
                logger.warning("[PostgresDB] Postgres not ready, waiting...")
                time.sleep(interval)

    def _get_db_schema(self):
        inspector = inspect(self.engine)
        schema = {}
        for table_name in inspector.get_table_names(schema='public'):
            columns = inspector.get_columns(table_name, schema='public')
            schema[table_name] = [col["name"] for col in columns]
        logger.info(f"[PostgresDB] Fetched DB schema: {schema}")
        return schema

    def _init_sql_chain(self):
        logger.info("[PostgresDB] Initializing SQLDatabaseChain with tables: %s", list(self.schema_context.keys()))
        sql_db = SQLDatabase.from_uri(
            self.db_url,
            include_tables=list(self.schema_context.keys()) or None
        )
        self.sql_chain = SQLDatabaseChain.from_llm(
            llm=self.llm,
            db=sql_db,
            verbose=True,
            return_intermediate_steps=False
        )

    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace"):
        logger.info(f"[PostgresDB] Inserting into {table_name} ({len(df)} rows)")
        try:
            df.to_sql(name=table_name, con=self.engine, if_exists=if_exists, index=False, method="multi")
            logger.info(f"[PostgresDB] Inserted {len(df)} rows into {table_name}")
        except SQLAlchemyError as e:
            logger.error("[PostgresDB] Insert failed: %s", e)
            raise

    def execute_query(self, query: str, fetch: bool = True):
        logger.info(f"[PostgresDB] Executing raw SQL: {query}")
        with self.engine.connect() as conn:
            result = conn.execute(sa.text(query))
            if fetch:
                return result.fetchall()

    def query_with_llm(self, user_query: str):
        logger.info("[PostgresDB] Refreshing schema before executing query...")

        # Refresh schema
        self.schema_context = self._get_db_schema()
        self._init_sql_chain()

        # Step 0: Extract main entity from NL query
        entity = extract_main_entity(user_query)
        if not entity:
            logger.warning("[PostgresDB] Could not extract entity from user query")
            return []

        logger.info(f"[PostgresDB] Querying DB for entity: {entity}")

        # Step 1: Generate SQL from NL query via LLM
        llm_result = self.sql_chain.invoke({"query": entity})

        # Extract the SQL only
        result_text = llm_result.get("result", "")
        match = re.search(r"SQLQuery:(.*)", result_text, re.IGNORECASE | re.DOTALL)
        if match:
            sql_query = match.group(1).strip()
        else:
            logger.warning("[PostgresDB] Could not find SQLQuery in LLM output")
            return []

        logger.info(f"[PostgresDB] Extracted SQL: {sql_query}")

        # Step 2: Execute SQL
        try:
            rows = self.execute_query(sql_query)
            # Convert to list of dicts for structured output
            return [dict(r._mapping) for r in rows]
        except Exception as e:
            logger.error(f"[PostgresDB] Failed to execute SQL: {e}")
            return []
