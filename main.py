
from __future__ import annotations

import argparse
import webbrowser


def run_flask_ui(host: str = "127.0.0.1", port: int = 5000, open_browser: bool = True):
    from flask_ui.app import create_app

    url = f"http://{host}:{port}"
    if open_browser:
        webbrowser.open(url)

    app = create_app()
    print(f"Movie RAG Flask UI: {url}")
    app.run(host=host, port=port, debug=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the movie RAG recommender.")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run the original terminal chat instead of the Flask UI.",
    )
    parser.add_argument("--query", default=None, help="Run a single terminal query.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1", help="Flask UI host.")
    parser.add_argument("--port", type=int, default=5000, help="Flask UI port.")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start Flask without opening the browser automatically.",
    )
    args = parser.parse_args()

    if args.cli or args.query:
        from rag.main import run_chat, run_once

        if args.query:
            run_once(args.query, args.top_k)
        else:
            run_chat()
        return

    run_flask_ui(args.host, args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
