from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

# Carrega variáveis do arquivo .env localizado na raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)

VALID_LLM_PROVIDERS = ("ollama", "gemini", "claude")
VALID_EMBEDDING_PROVIDERS = ("ollama", "gemini")

DEFAULT_LLM_MODELS: dict[str, str] = {
    "ollama": "llama3.2",
    "gemini": "gemini-3.8-flash",
    "claude": "claude-sonnet-4-20250514",
}

DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    "ollama": "nomic-embed-text",
    "gemini": "models/gemini-embedding-001",
}


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(
            f"Variável de ambiente obrigatória '{name}' não definida no arquivo .env."
        )
    return value


@dataclass(frozen=True)
class Settings:
    # Provedores (padrão: ollama)
    llm_provider: str
    embedding_provider: str

    # Modelos e temperatura
    llm_model: str
    llm_temperature: float
    embedding_model: str

    # Credenciais / endpoints dos provedores
    ollama_base_url: str
    google_api_key: str | None
    anthropic_api_key: str | None

    # Banco vetorial e RAG
    database_url: str
    collection_name: str
    top_k: int
    chunks_dir: Path
    rerank_top_n: int
    max_retries: int


def load_settings() -> Settings:
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if llm_provider not in VALID_LLM_PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER inválido: '{llm_provider}'. "
            f"Opções válidas: {', '.join(VALID_LLM_PROVIDERS)}."
        )

    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "ollama").strip().lower()
    if embedding_provider not in VALID_EMBEDDING_PROVIDERS:
        raise ValueError(
            f"EMBEDDING_PROVIDER inválido: '{embedding_provider}'. "
            f"Opções válidas: {', '.join(VALID_EMBEDDING_PROVIDERS)} "
            "(Claude/Anthropic não possui modelo próprio de embeddings)."
        )

    llm_model = (
        os.getenv("LLM_MODEL", "").strip() or DEFAULT_LLM_MODELS[llm_provider]
    )
    embedding_model = (
        os.getenv("EMBEDDING_MODEL", "").strip()
        or DEFAULT_EMBEDDING_MODELS[embedding_provider]
    )

    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip() or None
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip() or None

    if "gemini" in (llm_provider, embedding_provider) and not google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY é obrigatória quando LLM_PROVIDER ou EMBEDDING_PROVIDER é 'gemini'."
        )
    if llm_provider == "claude" and not anthropic_api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY é obrigatória quando LLM_PROVIDER é 'claude'."
        )

    chunks_dir_raw = os.getenv("CHUNKS_DIR", "chunks").strip()
    chunks_dir = Path(chunks_dir_raw)
    if not chunks_dir.is_absolute():
        chunks_dir = PROJECT_ROOT / chunks_dir

    return Settings(
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        llm_model=llm_model,
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        embedding_model=embedding_model,
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        ).strip(),
        google_api_key=google_api_key,
        anthropic_api_key=anthropic_api_key,
        database_url=_require_env("DATABASE_URL"),
        collection_name=os.getenv("COLLECTION_NAME", "politica_viagens").strip(),
        top_k=int(os.getenv("TOP_K", "4")),
        chunks_dir=chunks_dir,
        rerank_top_n=int(os.getenv("RERANK_TOP_N", "3")),
        max_retries=int(os.getenv("MAX_RETRIES", "2")),
    )


settings = load_settings()
