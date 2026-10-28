MIN_QUESTION_LENGTH = 3


def validate_question_payload(payload):
    """Validate the JSON body for POST /api/ask.

    Return:
        (question, None) when valid
        (None, error_dict) when invalid
    """
    def invalid(code, message):
        # Every error has a short code (for programs) and a message (for people)
        return None, {"error": code, "message": message}

    # The body must be a JSON object, like {"question": "..."}
    if not isinstance(payload, dict):
        return invalid("invalid_request", "Request body must be a JSON object.")

    # The question field must be there...
    if "question" not in payload:
        return invalid("missing_question", "The 'question' field is required.")

    question = payload["question"]

    # ...and it must be text
    if not isinstance(question, str):
        return invalid("invalid_question", "The 'question' field must be a string.")

    # Remove extra spaces/newlines from the ends
    question = question.strip()

    if not question:
        return invalid("empty_question", "The 'question' field cannot be blank.")

    if len(question) < MIN_QUESTION_LENGTH:
        return invalid(
            "short_question",
            f"The 'question' field must be at least {MIN_QUESTION_LENGTH} characters long.",
        )

    return question, None
