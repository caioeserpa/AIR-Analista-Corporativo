import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from assistente_corporativo.config import (
    DEFAULT_LLM_MODELS,
    VALID_LLM_PROVIDERS,
    settings,
)
from assistente_corporativo.retriever import RetrievedChunk, retrieve_chunks

logger = logging.getLogger(__name__)

NO_CONTEXT_RESPONSE = (
    "A documentação disponibilizada não contém informações suficientes para "
    "responder à sua pergunta. Por favor, consulte o setor responsável."
)

SYSTEM_PROMPT = """Você é um assistente corporativo especializado na política interna de viagens corporativas e reembolsos da empresa.

Siga rigorosamente estas regras:
1. Responda EXCLUSIVAMENTE com base nos trechos da documentação fornecidos no contexto.
2. NÃO invente regras, valores, prazos ou exceções, e NÃO utilize conhecimento externo.
3. Se o contexto fornecido não contiver informações suficientes para responder à pergunta, informe claramente que a documentação disponibilizada não contém informações sobre o assunto e que não é possível confirmar a regra.
4. Quando a pergunta envolver cálculos simples, prazos ou aplicação direta de uma regra presente no contexto (por exemplo, percentuais de adiantamento, diferenças a devolver ou contagem de dias úteis), apresente o cálculo ou raciocínio de forma clara e objetiva.
5. Seja claro, objetivo e coerente com a documentação, citando ao final ou ao longo da resposta os identificadores dos trechos utilizados (ex.: Chunk 006).
"""


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Formata os chunks recuperados em um bloco de contexto com identificação da origem."""
    formatted_blocks: list[str] = []
    for chunk in chunks:
        header = f"[Chunk {chunk.chunk_id} | {chunk.title} | fonte: {chunk.source}]"
        formatted_blocks.append(f"{header}\n{chunk.content}")
    return "\n\n---\n\n".join(formatted_blocks)


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """Factory do LLM: padrão 'ollama', alternativas 'gemini' e 'claude'."""
    selected_provider = (provider or settings.llm_provider).strip().lower()
    if selected_provider not in VALID_LLM_PROVIDERS:
        raise ValueError(
            f"Provedor de LLM inválido: '{selected_provider}'. "
            f"Use um de: {', '.join(VALID_LLM_PROVIDERS)}."
        )

    selected_model = (
        model
        or (
            settings.llm_model
            if selected_provider == settings.llm_provider
            else DEFAULT_LLM_MODELS[selected_provider]
        )
    )
    selected_temp = (
        temperature if temperature is not None else settings.llm_temperature
    )

    if selected_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=selected_model,
            temperature=selected_temp,
            base_url=settings.ollama_base_url,
        )

    if selected_provider == "gemini":
        if not settings.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY não configurada no .env para usar o Gemini."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=selected_temp,
            google_api_key=settings.google_api_key,
        )

    if selected_provider == "claude":
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY não configurada no .env para usar o Claude."
            )
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=selected_model,
            temperature=selected_temp,
            api_key=settings.anthropic_api_key,
        )

    raise ValueError(f"Provedor de LLM não suportado: '{selected_provider}'.")


def _extract_text(content: str | list) -> str:
    """Extrai texto do retorno do LLM (compatível com str ou lista de blocos)."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "".join(parts)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    provider: str | None = None,
    model: str | None = None,
    stream: bool = False,
) -> str:
    """Gera a resposta fundamentada exclusivamente nos chunks recuperados.

    Quando `stream=True`, imprime a resposta em tempo real conforme o modelo gera
    os tokens e ainda retorna o texto completo ao final.
    """
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("A pergunta não pode estar vazia.")

    if not chunks:
        logger.info(
            "Nenhum chunk fornecido; retornando resposta padrão de ausência de contexto."
        )
        if stream:
            print(NO_CONTEXT_RESPONSE, flush=True)
        return NO_CONTEXT_RESPONSE

    context = format_context(chunks)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Contexto da política de viagens corporativas:\n\n{context}\n\n"
                f"Pergunta do colaborador:\n{cleaned_question}"
            )
        ),
    ]

    selected_provider = (provider or settings.llm_provider).strip().lower()
    llm = get_llm(provider=selected_provider, model=model)

    if stream:
        collected: list[str] = []
        for token_chunk in llm.stream(messages):
            text = _extract_text(token_chunk.content)
            if text:
                print(text, end="", flush=True)
                collected.append(text)
        print()  # quebra de linha ao final do stream
        answer = "".join(collected).strip()
    else:
        response = llm.invoke(messages)
        answer = _extract_text(response.content).strip()

    logger.info(
        "Resposta gerada com %s ('%s') usando chunks: %s",
        selected_provider,
        model or settings.llm_model,
        [chunk.chunk_id for chunk in chunks],
    )
    return answer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    test_questions = [
        "Quais são os limites financeiros para hospedagem e alimentação durante uma viagem corporativa?",
        "Vou realizar uma viagem com despesas estimadas em R$ 4.000,00. Qual é o valor máximo que posso solicitar como adiantamento?",
        "A empresa permite que o colaborador leve um acompanhante durante a viagem corporativa?",
    ]

    for question in test_questions:
        print(f"\n{'=' * 80}\nPergunta: {question}\n{'=' * 80}")
        retrieved = retrieve_chunks(question)
        print(f"Chunks recuperados: {[c.chunk_id for c in retrieved]}\n\nResposta:")
        generate_answer(question, retrieved, stream=True)
        print()
