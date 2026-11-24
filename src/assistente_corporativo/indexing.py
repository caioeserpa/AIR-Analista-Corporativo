from pathlib import Path
import re

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

from assistente_corporativo.config import (
    DEFAULT_EMBEDDING_MODELS,
    VALID_EMBEDDING_PROVIDERS,
    settings,
)


def _slugify(value: str) -> str:
    cleaned = value.strip().lower().removeprefix("models/")
    return re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")


def resolve_collection_name(
    collection_name: str | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
) -> str:
    """Resolve o nome da coleção separada por provedor e modelo de embedding.

    Se `collection_name` for passado explicitamente, usa esse nome.
    Caso contrário, gera `<COLLECTION_NAME>__<provider>__<model>` para que
    múltiplos modelos (com dimensões diferentes) coexistam no PostgreSQL sem
    conflito nem necessidade de reindexar ao alternar de provedor.
    """
    if collection_name:
        return collection_name

    provider = (embedding_provider or settings.embedding_provider).strip().lower()
    model = (
        embedding_model
        or (
            settings.embedding_model
            if provider == settings.embedding_provider
            else DEFAULT_EMBEDDING_MODELS[provider]
        )
    )
    return f"{settings.collection_name}__{_slugify(provider)}__{_slugify(model)}"


def load_chunks(chunks_dir: Path | None = None) -> list[Document]:
    """Lê os arquivos .txt de chunks sem alterar o conteúdo original e extrai metadados."""
    directory = chunks_dir or settings.chunks_dir
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Pasta de chunks não encontrada: {directory}")

    chunk_files = sorted(directory.glob("*.txt"))
    if not chunk_files:
        raise ValueError(f"Nenhum arquivo .txt encontrado em: {directory}")

    documents: list[Document] = []
    for file_path in chunk_files:
        content = file_path.read_text(encoding="utf-8").strip()
        stem = file_path.stem

        if "_" in stem:
            chunk_id, raw_title = stem.split("_", 1)
            title = raw_title.replace("_", " ")
        else:
            chunk_id = stem
            title = stem

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "chunk_id": chunk_id,
                    "title": title,
                    "source": file_path.name,
                },
            )
        )

    return documents


def get_embeddings(
    provider: str | None = None,
    model: str | None = None,
) -> Embeddings:
    """Factory do modelo de embeddings: padrão 'ollama', alternativa 'gemini'."""
    selected_provider = (provider or settings.embedding_provider).strip().lower()
    if selected_provider not in VALID_EMBEDDING_PROVIDERS:
        raise ValueError(
            f"Provedor de embeddings inválido: '{selected_provider}'. "
            f"Use um de: {', '.join(VALID_EMBEDDING_PROVIDERS)}."
        )

    selected_model = (
        model
        or (
            settings.embedding_model
            if selected_provider == settings.embedding_provider
            else DEFAULT_EMBEDDING_MODELS[selected_provider]
        )
    )

    if selected_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=selected_model,
            base_url=settings.ollama_base_url,
        )

    if selected_provider == "gemini":
        if not settings.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY não configurada no .env para usar embeddings do Gemini."
            )
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=selected_model,
            google_api_key=settings.google_api_key,
        )

    raise ValueError(f"Provedor de embeddings não suportado: '{selected_provider}'.")


def get_vector_store(
    collection_name: str | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    pre_delete_collection: bool = False,
) -> PGVector:
    """Retorna a instância do banco vetorial PGVector conectada ao PostgreSQL."""
    embeddings = get_embeddings(
        provider=embedding_provider,
        model=embedding_model,
    )
    resolved_collection = resolve_collection_name(
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )
    return PGVector(
        embeddings=embeddings,
        collection_name=resolved_collection,
        connection=settings.database_url,
        use_jsonb=True,
        pre_delete_collection=pre_delete_collection,
    )


def index_chunks(
    chunks_dir: Path | None = None,
    collection_name: str | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
) -> PGVector:
    """Indexa todos os chunks no PGVector de forma idempotente."""
    documents = load_chunks(chunks_dir=chunks_dir)
    target_provider = (embedding_provider or settings.embedding_provider).strip().lower()
    target_model = (
        embedding_model
        or (
            settings.embedding_model
            if target_provider == settings.embedding_provider
            else DEFAULT_EMBEDDING_MODELS[target_provider]
        )
    )
    target_collection = resolve_collection_name(
        collection_name=collection_name,
        embedding_provider=target_provider,
        embedding_model=target_model,
    )

    vector_store = get_vector_store(
        collection_name=target_collection,
        embedding_provider=target_provider,
        embedding_model=target_model,
        pre_delete_collection=True,
    )

    # Prefixa o ID com o nome da coleção para evitar colisão de chave primária
    # entre coleções diferentes na tabela langchain_pg_embedding.
    ids = [f"{target_collection}:{doc.metadata['chunk_id']}" for doc in documents]
    vector_store.add_documents(documents, ids=ids)

    print(
        f"Indexação concluída: {len(documents)} chunks salvos na coleção "
        f"'{target_collection}' usando {target_provider} ('{target_model}')."
    )
    return vector_store


if __name__ == "__main__":
    store = index_chunks()

    # Teste rápido de busca semântica para validar a indexação
    test_query = "Qual é o limite diário de reembolso para despesas com alimentação?"
    print(f"\nTeste de busca: {test_query}\n")
    results = store.similarity_search_with_score(test_query, k=3)
    for rank, (doc, score) in enumerate(results, start=1):
        print(
            f"{rank}. [Chunk {doc.metadata['chunk_id']} - {doc.metadata['title']}] "
            f"(distância: {score:.4f})"
        )
        print(f"   {doc.page_content}\n")
