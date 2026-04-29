

from __future__ import annotations

import gradio as gr

from .main import create_pipeline


pipeline = create_pipeline()


def answer(query: str, top_k: int) -> str:
    response = pipeline.query(query, top_k=top_k)
    retrieved = "\n".join(
        f"- {result['Title']} ({result['Distance']:.4f})"
        for result in response["results"]
    )
    return f"{response['answer']}\n\nRetrieved:\n{retrieved}"


demo = gr.Interface(
    fn=answer,
    inputs=[
        gr.Textbox(lines=3, label="Movie query"),
        gr.Slider(1, 10, value=5, step=1, label="Top K"),
    ],
    outputs=gr.Textbox(lines=18, label="Answer"),
    title="Movie RAG Recommender",
)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7863)
