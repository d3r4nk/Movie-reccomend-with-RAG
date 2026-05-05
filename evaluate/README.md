# Movie RAG Evaluation

This folder evaluates the movie RAG pipeline implemented in `rag/`.

## Run

```powershell
python -m evaluate.run_evaluation --sample-size 14 --top-k 5
```

The evaluator reads `OPENAI_API_KEY` from the environment or from `.env`.
The judge model defaults to `gpt-5.5` unless `OPENAI_EVAL_MODEL` or
`--openai-judge-model` is set.

## Evaluation Flow

1. Load `data/processed/IMDb_Dataset_Composite_Cleaned.csv`.
2. Create a holdout source split for generating evaluation cases.
3. Write `train_split.csv`, `eval_case_source.csv`, and `indexed_corpus.csv`.
4. Build `full_movie_chunks_metadata.csv` from the indexed corpus.
5. Build a temporary ChromaDB index under the run output directory.
6. Run `MovieRAGPipeline.query()` for every generated case.
7. Compute exact-title retrieval diagnostics from retrieved titles.
8. Send query, retrieved context, and RAG answer to the OpenAI judge.
9. Write JSON, CSV, ChromaDB files, and PNG dashboards under `evaluate/results/<run-id>/`.

Source: [`run_evaluation.py`](run_evaluation.py).

## Use Cases

The evaluator rotates through these use-case templates:

| Use case | Query shape |
| --- | --- |
| `genre_rating` | Genre request constrained by IMDb rating phrase. |
| `director_year` | Similar movie request using genre, director, and approximate year. |
| `actor_genre` | Genre request using one actor or a similar cast profile. |
| `duration_certificate` | Genre request constrained by certificate and runtime phrase. |
| `multi_constraint` | Combined genres, approximate year, rating phrase, and director preference. |
| `vague_preference` | Broad similarity request using genre and year period. |
| `out_of_scope` | Unsupported query requiring the model to say context is insufficient. |

Source: `make_query()` and `make_eval_cases()` in [`run_evaluation.py`](run_evaluation.py).

## Metric Groups

### Judge Metrics

These metrics are scored by the OpenAI judge as integers from `1` to `5`:

| Metric | Meaning in this evaluator |
| --- | --- |
| `accuracy` | Whether recommendations satisfy the query, not whether they match `reference_title`. |
| `faithfulness` | Whether the answer is grounded in retrieved context. |
| `answer_relevance` | Whether the answer addresses the query. |
| `context_relevance` | Whether retrieved context is relevant to the query. |
| `semantic_context_precision` | How much of `retrieved_context` is relevant to the query. |
| `semantic_context_recall` | How completely `retrieved_context` covers the query intent and constraints. |

The dashboard normalizes these 1-5 scores to 0-1 with:

```text
normalized_score = raw_score / 5
```

`mean_context_precision` is therefore:

```text
mean(score_ratio(semantic_context_precision))
```

It is a semantic context precision score judged by an LLM. It is not the
traditional retrieval precision formula.

### Semantic Retrieval Metrics

These metrics are also returned by the judge:

| Metric | Meaning |
| --- | --- |
| `retrieval_hit_rate` | Mean of `semantic_retrieval_hit`; true when at least one retrieved movie is semantically relevant. |
| `retrieval_top1_rate` | Mean of `semantic_retrieval_top1`; true when the first retrieved movie is semantically relevant. |
| `mean_reciprocal_rank` | Mean of `semantic_reciprocal_rank`; reciprocal rank of the first semantically relevant retrieved movie. |

### Exact-Title Retrieval Diagnostics

These metrics are computed directly from retrieved titles and `expected_title`:

| Metric | Formula |
| --- | --- |
| `exact_title_hit_rate` | Mean of whether `expected_title` appears in retrieved titles. |
| `exact_title_top1_rate` | Mean of whether rank 1 equals `expected_title`. |
| `mean_exact_title_reciprocal_rank` | Mean of `1 / exact_title_rank`, or `0` if not retrieved. |
| `exact_title_precision` | `relevant_retrieved_count / total_retrieved_count`. |
| `exact_title_recall` | `relevant_retrieved_count / total_relevant_count`. |

For this dataset shape, `relevant_retrieved_count` is the number of retrieved
chunks whose normalized title equals `expected_title`.

### Distance And Reranking Metrics

| Metric | Meaning |
| --- | --- |
| `mean_avg_distance` | Mean ChromaDB distance across final top-k retrieved chunks. |
| `mean_best_distance` | Mean of the best ChromaDB distance in each final top-k result set. |
| `mean_top_distance` | Mean ChromaDB distance of the top-ranked final chunk. |
| `mean_avg_rerank_score` | Mean cross-encoder rerank score across final top-k chunks. |
| `mean_top_rerank_score` | Mean rerank score of the top-ranked final chunk. |
| `mean_best_rerank_score` | Mean best rerank score in each final top-k result set. |

### Quality And Failure Metrics

| Metric | Meaning |
| --- | --- |
| `rag_score` | Mean of normalized `accuracy`, `faithfulness`, `answer_relevance`, `context_relevance`, `semantic_context_precision`, and `semantic_context_recall`. |
| `pass_rate` | Share of cases whose `rag_score` meets `--pass-threshold` and are not marked hallucinated or wrong-context. |
| `hallucination_rate` | Share of cases where the judge marks unsupported answer content. |
| `wrong_context_rate` | Share of cases where `uses_correct_context` is false. |
| `metadata_filter_applied_rate` | Share of cases where structured Chroma metadata filtering was applied. |
| `generation_error_count` | Number of cases with RAG generation errors. |
| `judge_error_count` | Number of cases with judge errors. |

## Output Files

Each run writes to:

```text
evaluate/results/<run-id>/
```

| File or folder | Meaning |
| --- | --- |
| `summary.json` | Aggregate metrics, run metadata, model names, use-case counts, and output paths. |
| `results.json` | Full per-case records including query, answer, retrieved results, judge fields, filters, and errors. |
| `results.csv` | Flat per-case table for spreadsheet inspection. |
| `train_split.csv` | Non-evaluation source rows from the holdout split. |
| `eval_case_source.csv` | Source rows used to generate evaluation cases. |
| `indexed_corpus.csv` | Full filtered corpus used to build the evaluation vector index. |
| `full_movie_chunks_metadata.csv` | Movie chunks used to build the evaluation vector index. |
| `chroma_full/` | Temporary persistent ChromaDB index for the run. |
| `dashboard.png` | Aggregate dashboard image. |
| `metric_bars.png` | Bar chart for selected aggregate metrics. |
| `cases.png` | Per-case visual table. |

## Saved Run: `20260505-130313`

The saved run at `evaluate/results/20260505-130313/summary.json` contains:

| Field | Value |
| --- | --- |
| `case_count` | `14` |
| `top_k` | `5` |
| `rag_llm_model` | `qwen2.5-7b-instruct` |
| `embedding_model` | `sentence-transformers/all-mpnet-base-v2` |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `openai_judge_model` | `gpt-5.5` |
| `rag_score` | `0.838095238095238` |
| `pass_rate` | `0.6428571428571429` |
| `retrieval_hit_rate` | `0.8571428571428571` |
| `retrieval_top1_rate` | `0.8571428571428571` |
| `mean_context_precision` | `0.8285714285714285` |
| `mean_context_recall` | `0.7857142857142857` |
| `hallucination_rate` | `0.21428571428571427` |
| `metadata_filter_applied_rate` | `0.7857142857142857` |

![Evaluation dashboard](results/20260505-130313/dashboard.png)

## Project Images

The root `images/` folder contains these PNG references:

![System architecture](../images/system_architecture.png)

![End-to-end flow](../images/end_to_end_flow.png)

![ChromaDB storage](../images/chromadb_storage.png)

![Top-k ChromaDB query](../images/topk_chromadb_query.png)

![Movie recommendation UI result](../images/rec_result.png)
