import inspect

from lib.model_client import ModelClientError, generate_answer
from lib.prompt_builder import build_rag_prompt
from lib.response_formatter import (
    format_fallback_response,
    format_sources,
    format_success_response,
)
from lib.retrieval import DEFAULT_TOP_K, retrieve_context


def _accepts_top_k(function):
    """True if the function can be called with a top_k argument."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        p.name == "top_k" or p.kind is inspect.Parameter.VAR_KEYWORD
        for p in parameters
    )


def answer_question(
    question,
    retriever=None,
    prompt_builder=None,
    model_client=None,
    top_k=DEFAULT_TOP_K,
):
    """Run the full RAG workflow for a validated question."""
    # Use whatever was handed in (tests do this), otherwise the real helpers
    retriever = retriever or retrieve_context
    prompt_builder = prompt_builder or build_rag_prompt
    model_client = model_client or generate_answer

    # 1. Retrieve context first
    if _accepts_top_k(retriever):
        chunks = retriever(question, top_k=top_k)
    else:
        chunks = retriever(question)

    # 2. No useful context? Don't bother the model at all
    usable_chunks = [
        chunk for chunk in (chunks or [])
        if isinstance(chunk, dict)
        and isinstance(chunk.get("text"), str)
        and chunk["text"].strip()
    ]
    if not usable_chunks:
        return format_fallback_response(question)

    # 3. Build the prompt, 4. ask the model
    prompt = prompt_builder(question, usable_chunks)
    answer = model_client(prompt)

    # 5. A blank or odd answer is a model problem, not a success
    if not isinstance(answer, str) or not answer.strip():
        raise ModelClientError("Model returned an empty or unusable answer.")

    # 6. Answer plus where it came from
    return format_success_response(answer, format_sources(usable_chunks))
