from flask import Flask, jsonify, request

from lib.model_client import ModelClientError
from lib.rag_service import answer_question
from lib.response_formatter import format_error_response
from lib.validation import validate_question_payload


def create_app():
    app = Flask(__name__)

    @app.post("/api/ask")
    def ask():
        # Read the JSON body (None if it's missing or not valid JSON)
        payload = request.get_json(silent=True)

        # Check it; bad input gets a 400
        question, error = validate_question_payload(payload)
        if error:
            return jsonify(error), 400

        # Hand the real work to the RAG service
        try:
            result = answer_question(question)
        except ModelClientError as error:
            # The model service failed, so 502 (bad gateway), not our fault
            return jsonify(format_error_response(
                "model_service_error", str(error),
            )), 502

        # Success and "not enough info" fallback both come back as 200
        return jsonify(result), 200

    return app


app = create_app()
