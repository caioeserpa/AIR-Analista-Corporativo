from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

# Carrega variáveis do arquivo .env localizado na raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(
            f"Variável de ambiente obrigatória '{name}' não definida no arquivo .env."
        )
    return value


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    llm_model: str
    llm_temperature: float
    embedding_model: str
    database_url: str
    collection_name: str
    top_k: int
    chunks_dir: Path
    rerank_top_n: int
    max_retries: int


def load_settings() -> Settings:
    chunks_dir_raw = os.getenv("CHUNKS_DIR", "chunks").strip()
    chunks_dir = Path(chunks_dir_raw)
    if not chunks_dir.is_absolute():
        chunks_dir = PROJECT_ROOT / chunks_dir

    return Settings(
        google_api_key=_require_env("GOOGLE_API_KEY"),
        llm_model=os.getenv("LLM_MODEL", "gemini-2.5-flash").strip(),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "models/gemini-embedding-001"
        ).strip(),
        database_url=_require_env("DATABASE_URL"),
        collection_name=os.getenv("COLLECTION_NAME", "politica_viagens").strip(),
        top_k=int(os.getenv("TOP_K", "4")),
        chunks_dir=chunks_dir,
        rerank_top_n=int(os.getenv("RERANK_TOP_N", "3")),
        max_retries=int(os.getenv("MAX_RETRIES", "2")),
    )


settings = load_settings()
