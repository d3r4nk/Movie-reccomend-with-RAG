from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, render_template, request


pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        from rag.main import create_pipeline

        pipeline = create_pipeline()
    return pipeline


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/query")
    def query():
        payload: dict[str, Any] = request.get_json(silent=True) or {}
        user_query = str(payload.get("query", "")).strip()

        try:
            top_k = int(payload.get("top_k", 5))
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(top_k, 10))

        if not user_query:
            return jsonify({"answer": "Please enter a movie query.", "results": []})

        try:
            response = get_pipeline().query(user_query, top_k=top_k)
        except Exception as exc:
            return (
                jsonify(
                    {
                        "error": (
                            "Query failed. Make sure ChromaDB is built and LM Studio "
                            "is running at the configured OpenAI-compatible endpoint."
                        ),
                        "details": str(exc),
                    }
                ),
                500,
            )

        return jsonify(response)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
