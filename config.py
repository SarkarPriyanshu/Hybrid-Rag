import os
from urllib.parse import quote_plus
from typing import Optional, Dict, Any

# -----------------------------
# General Configuration
# -----------------------------
class GeneralConfig:
    def __init__(self):
        self.base_url: str = os.getenv(
            "BASE_DAILY_MED_URL", 
            "https://dailymed.nlm.nih.gov/dailymed/services/v2"
        )
        self.max_rows: int = int(os.getenv("MAX_ROWS", 1000))
        self.page_size: int = int(os.getenv("PAGE_SIZE", 5))
        self.page_no: int = int(os.getenv("PAGE_NO", 1))
        
        # Use a consistent path
        self.data_folder: str = os.getenv("DATA_FOLDER", "app/data")
        os.makedirs(self.data_folder, exist_ok=True)
        self.csv_path: str = os.path.join(self.data_folder, "dailymed_medicines.csv")


# -----------------------------
# Kafka Configuration
# -----------------------------
class KafkaConfig:
    def __init__(self):
        self.broker_url: str = os.getenv("KAFKA_BROKER_URL", "localhost:9092")
        self.topic: str = os.getenv("KAFKA_TOPIC", "rag-topic")
        self.group_id: str = os.getenv("KAFKA_GROUP_ID", "rag-group")


# -----------------------------
# LLM / LangChain Configuration
# -----------------------------
class LLMConfig:
    def __init__(self):
        self.provider: str = os.getenv("LLM_PROVIDER", "gemini")
        self.api_key: Optional[str] = os.getenv("LLM_API_KEY", "")
        self.model_name: str = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")


# -----------------------------
# Database Configuration
# -----------------------------
class DBConfig:
    def __init__(self):
        # Host/Port/DB credentials
        self.host: str = os.getenv("DB_HOST", "db")          # Docker service name
        self.port: int = int(os.getenv("DB_PORT", 5432))     # PostgreSQL default
        self.db_name: str = os.getenv("DB_NAME", "hybrid_rag")
        self.username: str = os.getenv("DB_USER", "postgres")
        self.password: str = os.getenv("DB_PASSWORD", "StrongPassw0rd!")
        self.db_table_name: str = os.getenv("DB_TABLE_NAME", "medicines")

        # URL encoding (in case password has special chars)
        encoded_password = quote_plus(self.password)

        # Construct DB URL (psycopg2 driver)
        self.db_url: str = os.getenv("DB_URL")
        if not self.db_url:
            self.db_url = (
                f"postgresql+psycopg://{self.username}:{encoded_password}"
                f"@{self.host}:{self.port}/{self.db_name}"
            )

        # Optional timeout setting
        self.connect_timeout: int = int(os.getenv("DB_CONNECT_TIMEOUT", 30))

# -----------------------------
# VectorDB Configuration
# -----------------------------
class VectorDBConfig:
    def __init__(self):
        self.provider: str = os.getenv("VECTOR_DB_PROVIDER", "pinecone")
        self.vector_db_api_key: str = os.getenv("VECTOR_DB_API_KEY", "")
        self.index_path: str = os.getenv("VECTOR_INDEX_PATH", "./data/index.faiss")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


# -----------------------------
# Main Config Wrapper
# -----------------------------
class Config:
    def __init__(self):
        self.env: str = os.getenv("APP_ENV", "development")
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.port: int = int(os.getenv("PORT", 8000))
        self.components: Dict[str, Any] = {
            "general": GeneralConfig(),
            "kafka": KafkaConfig(),
            "llm": LLMConfig(),
            "db": DBConfig(),
            "vectordb": VectorDBConfig(),
        }


config = Config()
