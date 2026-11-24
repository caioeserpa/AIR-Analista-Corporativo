from dataclasses import asdict, dataclass
import logging

from assistente_corporativo.config import settings
from assistente_corporativo.indexing import get_vector_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    title: str
    source: str
    content: str
    score: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


def retrieve_chunks(
    query: str,
    top_k: int | None = None,
    collection_name: str | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
) -> list[RetrievedChunk]:
    """Busca os Top-K chunks mais relevantes para a pergunta no PGVector."""
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("A pergunta para busca não pode estar vazia.")

    k = top_k if top_k is not None else settings.top_k
    if k <= 0:
        raise ValueError("O parâmetro top_k deve ser maior que zero.")

    vector_store = get_vector_store(
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        pre_delete_collection=False,
    )

    raw_results = vector_store.similarity_search_with_score(cleaned_query, k=k)

    chunks: list[RetrievedChunk] = []
    for doc, score in raw_results:
        metadata = doc.metadata or {}
        chunks.append(
            RetrievedChunk(
                chunk_id=str(metadata.get("chunk_id", "")),
                title=str(metadata.get("title", "")),
                source=str(metadata.get("source", "")),
                content=doc.page_content,
                score=float(score),
            )
        )

    summary = ", ".join(
        f"{chunk.chunk_id} ({chunk.score:.4f})" for chunk in chunks
    )
    logger.info(
        "Busca concluída (top_k=%d) | Pergunta: '%s' | Chunks recuperados: [%s]",
        k,
        cleaned_query,
        summary,
    )

    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    test_queries = [
        "Quais são os limites financeiros para hospedagem e alimentação durante uma viagem corporativa?",
        "A empresa permite que o colaborador leve um acompanhante durante a viagem corporativa?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 80}\nPergunta: {query}\n{'=' * 80}")
        retrieved = retrieve_chunks(query)
        for rank, chunk in enumerate(retrieved, start=1):
            print(
                f"{rank}. [Chunk {chunk.chunk_id} - {chunk.title}] "
                f"(distância: {chunk.score:.4f} | fonte: {chunk.source})"
            )
            print(f"   {chunk.content}\n")
