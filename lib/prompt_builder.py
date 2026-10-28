def format_context_chunk(chunk):
    """Format one retrieved chunk for the prompt context block."""
    return (
        f"Source ID: {chunk.get('source_id')}\n"
        f"Title: {chunk.get('title')}\n"
        f"Category: {chunk.get('category')}\n"
        f"Section: {chunk.get('section')}\n"
        f"Text: {(chunk.get('text') or '').strip()}"
    )


def build_rag_prompt(question, context_chunks):
    """Build a structured RAG prompt from a question and retrieved context."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    # Only chunks that actually have text are useful context
    usable_chunks = [
        chunk for chunk in (context_chunks or [])
        if isinstance(chunk, dict)
        and isinstance(chunk.get("text"), str)
        and chunk["text"].strip()
    ]
    if not usable_chunks:
        raise ValueError("At least one context chunk with text is required.")

    context_block = "\n\n".join(format_context_chunk(chunk) for chunk in usable_chunks)

    return (
        "Instructions:\n"
        "You are an assistant that helps Customer Success support representatives. "
        "Use only the approved context below to answer the question. "
        "Do not invent details or make claims the context does not support.\n"
        "\n"
        "Context:\n"
        f"{context_block}\n"
        "\n"
        "Question:\n"
        f"{question.strip()}\n"
        "\n"
        "Response Requirements:\n"
        "- Answer using only the approved context.\n"
        "- If the context does not contain enough information, say what is missing.\n"
        "- Do not make up policies, numbers, or steps.\n"
        "- Keep the answer concise.\n"
        "- Refer to source IDs (such as SUB-101) when helpful.\n"
    )
