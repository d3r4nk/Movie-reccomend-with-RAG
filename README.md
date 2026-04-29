# RAG Movie Recommender

A local Retrieval-Augmented Generation movie recommender built with an IMDb dataset, ChromaDB vector search, SentenceTransformers embeddings, and **Qwen 2.5 7B Instruct** served through LM Studio.

## Preview

![Movie recommendation UI result](images/rec_result.png)

## Structure

```text
rag-movie-rec-chroma/
  data/
    crawl_data.py          # Download parthdande/imdb-dataset-2024-updated
    preprocess_data.py     # Merge, clean, describe, and chunk movie data
  rag/
    config.py              # Runtime settings
    embeddings.py          # Local SentenceTransformers embeddings
    chroma_store.py        # ChromaDB build/search wrapper
    build_vector_db.py     # Embed chunks and persist them in ChromaDB
    rag_pipeline.py        # Retrieval + generation chain
    main.py                # CLI chat/query runner
    app.py                 # Optional Gradio UI
  main.py                  # Root Flask UI entry point
  requirements.txt
```

## CSV Schemas

The preprocessing pipeline produces these files:

`IMDb_Dataset_Composite_Cleaned.csv`

```text
Title, IMDb Rating, Year, Certificates, Director, Star Cast,
MetaScore, Duration (minutes), Poster-src, Genres
```

`Movie_Descriptions.csv`

```text
Title, Description
```

`movie_chunks_metadata.csv`

```text
Title, Chunk, Metadata
```

## Preprocessing

The preprocessing pipeline:

- Download Kaggle dataset `parthdande/imdb-dataset-2024-updated`
- Read `IMDb_Dataset.csv`, `IMDb_Dataset_2.csv`, `IMDb_Dataset_3.csv`
- Drop duplicate titles per source file with `keep="last"`
- Merge by `Title`, updating older records with later files
- Fill missing `Poster-src` with `No Poster Available`
- Fill missing second/third genres with empty strings
- Combine `Genre`, `Second_Genre`, `Third_Genre` into `Genres`
- Clean `Star Cast` delimiters with regex-based normalization
- Generate one movie description per row
- Chunk each description with:

```python
RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
)
```

So chunking is per movie description, by character count, not by token.

## Setup

Use a project virtual environment so these dependencies do not overwrite packages
from another Python project:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

You need Kaggle credentials for crawling:

```text
%USERPROFILE%\.kaggle\kaggle.json
```

## Build Data

```bash
python -m data.crawl_data
python -m data.preprocess_data
```

This writes processed CSVs to:

```text
data/processed/
```

## Build ChromaDB

```bash
python -m rag.build_vector_db
```

This embeds `movie_chunks_metadata.csv` and persists ChromaDB to:

```text
chroma_db/
```

## Run LM Studio

In LM Studio:

1. Download/load **Qwen 2.5 7B Instruct**.
2. Start the local server.
3. Use the default OpenAI-compatible endpoint:

```text
http://localhost:1234/v1
```

You can override settings with environment variables:

```bash
set LM_STUDIO_BASE_URL=http://localhost:1234/v1
set LM_STUDIO_MODEL=qwen2.5-7b-instruct
set LM_STUDIO_API_KEY=lm-studio
```

## Query

Start the Flask UI:

```powershell
python main.py
```

Then open:

```text
http://127.0.0.1:5000
```

Single terminal query:

```bash
python main.py --query "Recommend smart action movies with high ratings"
```

Interactive terminal chat:

```bash
python main.py --cli
```

Optional Gradio UI:

```bash
python -m rag.app
```

You can still run Flask directly:

```powershell
python -m flask --app flask_ui.app run --host 127.0.0.1 --port 5000
```

## Notes

- Generation is local through LM Studio.
- Embeddings are local through `sentence-transformers/all-MiniLM-L6-v2`.
- ChromaDB stores the actual chunk text as documents and the movie title as metadata.
- If you want stronger retrieval embeddings later, change `EMBEDDING_MODEL_NAME`.
