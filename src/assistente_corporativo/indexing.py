from pathlib import Path

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

from assistente_corporativo.config import settings


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


def get_embeddings(model: str | None = None) -> GoogleGenerativeAIEmbeddings:
    """Instancia o modelo de embeddings do Google Gemini."""
    return GoogleGenerativeAIEmbeddings(
        model=model or settings.embedding_model,
        google_api_key=settings.google_api_key,
    )


def get_vector_store(
    collection_name: str | None = None,
    embedding_model: str | None = None,
    pre_delete_collection: bool = False,
) -> PGVector:
    """Retorna a instância do banco vetorial PGVector conectada ao PostgreSQL."""
    embeddings = get_embeddings(model=embedding_model)
    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name or settings.collection_name,
        connection=settings.database_url,
        use_jsonb=True,
        pre_delete_collection=pre_delete_collection,
    )


def index_chunks(
    chunks_dir: Path | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
) -> PGVector:
    """Indexa todos os chunks no PGVector de forma idempotente."""
    documents = load_chunks(chunks_dir=chunks_dir)
    target_collection = collection_name or settings.collection_name
    target_model = embedding_model or settings.embedding_model

    vector_store = get_vector_store(
        collection_name=target_collection,
        embedding_model=target_model,
        pre_delete_collection=True,
    )

    ids = [doc.metadata["chunk_id"] for doc in documents]
    vector_store.add_documents(documents, ids=ids)

    print(
        f"Indexação concluída: {len(documents)} chunks salvos na coleção "
        f"'{target_collection}' usando '{target_model}'."
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
