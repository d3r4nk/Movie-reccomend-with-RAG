from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]
EVALUATE_DIR = ROOT_DIR / "evaluate"
RESULTS_DIR = EVALUATE_DIR / "results"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.chroma_store import MovieChromaStore
from rag.config import RAGConfig
from rag.rag_pipeline import MovieRAGPipeline
from data.preprocess_data import chunk_descriptions, create_descriptions


@dataclass
class EvalCase:
    case_id: str
    query: str
    expected_title: str
    use_case: str
    source_fields: dict[str, Any]


def load_dotenv(path: Path = ROOT_DIR / ".env") -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def normalized_title(value: Any) -> str:
    return str(value or "").strip().casefold()


def safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp_score(value: Any, minimum: int = 1, maximum: int = 5) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = minimum
    return max(minimum, min(maximum, score))


def score_ratio(value: float | int | None, scale: float = 5.0) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value) / scale))


def clamp_ratio(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def optional_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def reciprocal_rank_from_judge(parsed: dict[str, Any]) -> float:
    explicit = parsed.get("semantic_reciprocal_rank")
    if explicit is not None:
        return clamp_ratio(explicit)
    rank = optional_positive_int(parsed.get("semantic_retrieval_rank"))
    return 0.0 if rank is None else 1.0 / rank


def load_relevant_chunk_counts(chunks_path: Path) -> dict[str, int]:
    if not chunks_path.exists():
        return {}

    chunks_df = pd.read_csv(chunks_path)
    if "Title" not in chunks_df.columns:
        return {}

    counts = chunks_df["Title"].astype(str).map(normalized_title).value_counts()
    return {str(title): int(count) for title, count in counts.items()}


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def first_list_item(value: Any) -> str:
    parts = [part.strip() for part in clean_text(value).split(",") if part.strip()]
    return parts[0] if parts else clean_text(value)


def split_people(value: Any) -> list[str]:
    return [part.strip() for part in clean_text(value).split(",") if part.strip()]


def rating_phrase(value: Any) -> str:
    rating = safe_float(value)
    if rating is None:
        return "with a documented IMDb rating"
    if rating >= 8:
        return "with an excellent IMDb rating"
    if rating >= 7:
        return "with a strong IMDb rating"
    if rating >= 6:
        return "with a moderate IMDb rating"
    return "with a lower IMDb rating"


def duration_phrase(value: Any) -> str:
    minutes = safe_float(value)
    if minutes is None:
        return "with a documented runtime"
    if minutes < 90:
        return "under 90 minutes"
    if minutes <= 120:
        return "around two hours or less"
    return "longer than two hours"


def load_filtered_dataset(dataset_path: Path, min_rating: float | None) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    required = {
        "Title",
        "IMDb Rating",
        "Year",
        "Certificates",
        "Director",
        "Star Cast",
        "MetaScore",
        "Duration (minutes)",
        "Poster-src",
        "Genres",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")

    df = df.dropna(subset=sorted(required))
    if min_rating is not None:
        df = df[df["IMDb Rating"].apply(lambda value: (safe_float(value) or 0) >= min_rating)]

    if df.empty:
        raise ValueError("No rows available for evaluation after filtering.")

    return df


def build_train_chunks(train_df: pd.DataFrame, chunks_path: Path) -> None:
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    descriptions_df = create_descriptions(train_df)
    chunks_df = chunk_descriptions(descriptions_df)
    chunks_df[["Title", "Chunk", "Metadata"]].to_csv(chunks_path, index=False)


def make_holdout_split(
    df: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sample_size < 1:
        raise ValueError("--sample-size must be at least 1")
    if len(df) <= sample_size:
        raise ValueError(
            "Dataset must contain more rows than --sample-size to create a non-overlapping train/eval split."
        )

    indexed_df = df.reset_index(drop=True).copy()
    indexed_df.insert(0, "source_row_id", range(len(indexed_df)))
    eval_case_df = indexed_df.sample(n=sample_size, random_state=seed)
    train_df = indexed_df.drop(index=eval_case_df.index)
    if set(train_df["source_row_id"]).intersection(eval_case_df["source_row_id"]):
        raise ValueError("Train/eval split overlap detected.")
    return train_df.reset_index(drop=True), eval_case_df.reset_index(drop=True)


def make_query(row: pd.Series, use_case: str) -> str:
    title = clean_text(row["Title"])
    year = clean_text(row["Year"])
    rating = clean_text(row["IMDb Rating"])
    director = clean_text(row["Director"])
    genres = clean_text(row["Genres"])
    primary_genre = first_list_item(genres)
    actors = split_people(row["Star Cast"])
    actor = actors[0] if actors else ""
    certificate = clean_text(row["Certificates"])
    duration = clean_text(row["Duration (minutes)"])

    if use_case == "genre_rating":
        return f"Recommend a {primary_genre} movie {rating_phrase(rating)}."
    if use_case == "director_year":
        return f"Suggest movies like a {primary_genre} film directed by {director} around {year}."
    if use_case == "actor_genre":
        return f"I want a {primary_genre} movie featuring {actor} or a similar cast profile."
    if use_case == "duration_certificate":
        return f"Recommend a {primary_genre} movie rated {certificate} and {duration_phrase(duration)}."
    if use_case == "multi_constraint":
        return (
            f"Find a {genres} movie from around {year}, {rating_phrase(rating)}, "
            f"preferably directed by someone like {director}."
        )
    if use_case == "vague_preference":
        return f"Recommend something similar in profile to a {primary_genre} movie from the {year} period."
    if use_case == "out_of_scope":
        return (
            "Recommend a movie about quantum cooking competitions with verified plot details. "
            "If the retrieved context does not support that, say the context is insufficient."
        )
    raise ValueError(f"Unknown use case: {use_case}")


def make_eval_cases(
    eval_df: pd.DataFrame,
    sample_size: int,
    seed: int,
    split_label: str = "full_index",
) -> list[EvalCase]:
    if eval_df.empty:
        raise ValueError("No rows available for evaluation.")

    sampled = eval_df.sample(n=min(sample_size, len(eval_df)), random_state=seed)
    use_cases = [
        "genre_rating",
        "director_year",
        "actor_genre",
        "duration_certificate",
        "multi_constraint",
        "vague_preference",
        "out_of_scope",
    ]
    cases: list[EvalCase] = []

    for index, row in sampled.reset_index(drop=True).iterrows():
        title = str(row["Title"]).strip()
        year = str(row["Year"]).strip()
        rating = str(row["IMDb Rating"]).strip()
        director = str(row["Director"]).strip()
        genres = str(row["Genres"]).strip()
        use_case = use_cases[index % len(use_cases)]
        query = make_query(row, use_case)
        cases.append(
            EvalCase(
                case_id=f"case-{index + 1:03d}",
                query=query,
                expected_title=title,
                use_case=use_case,
                source_fields={
                    "Title": title,
                    "Year": year,
                    "IMDb Rating": rating,
                    "Director": director,
                    "Star Cast": str(row["Star Cast"]).strip(),
                    "Certificates": str(row["Certificates"]).strip(),
                    "Duration (minutes)": str(row["Duration (minutes)"]).strip(),
                    "Genres": genres,
                    "split": split_label,
                },
            )
        )

    return cases


def retrieval_metrics(
    results: list[dict[str, Any]],
    expected_title: str,
    relevant_chunk_counts: dict[str, int],
) -> dict[str, Any]:
    expected = normalized_title(expected_title)
    retrieved_titles = [result.get("Title") for result in results]
    normalized = [normalized_title(title) for title in retrieved_titles]
    relevant_retrieved_count = sum(1 for title in normalized if title == expected)
    total_retrieved_count = len(results)
    total_relevant_count = relevant_chunk_counts.get(expected)

    rank = None
    for idx, title in enumerate(normalized, start=1):
        if title == expected:
            rank = idx
            break

    distances = [
        float(result["Distance"])
        for result in results
        if result.get("Distance") is not None
    ]
    rerank_scores = [
        float(result["RerankScore"])
        for result in results
        if result.get("RerankScore") is not None
    ]

    return {
        "exact_title_hit": rank is not None,
        "hit": rank is not None,
        "reference_title_retrieved": rank is not None,
        "exact_title_rank": rank,
        "rank": rank,
        "exact_title_reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
        "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
        "exact_title_top1_match": bool(rank == 1),
        "top1_match": bool(rank == 1),
        "exact_title_precision": (
            None
            if total_retrieved_count == 0
            else relevant_retrieved_count / total_retrieved_count
        ),
        "exact_title_recall": (
            None
            if not total_relevant_count
            else relevant_retrieved_count / total_relevant_count
        ),
        "relevant_retrieved_count": relevant_retrieved_count,
        "total_retrieved_count": total_retrieved_count,
        "total_relevant_count": total_relevant_count,
        "retrieved_titles": retrieved_titles,
        "avg_distance": None if not distances else sum(distances) / len(distances),
        "best_distance": None if not distances else min(distances),
        "top_distance": distances[0] if distances else None,
        "avg_rerank_score": None
        if not rerank_scores
        else sum(rerank_scores) / len(rerank_scores),
        "top_rerank_score": rerank_scores[0] if rerank_scores else None,
        "best_rerank_score": None if not rerank_scores else max(rerank_scores),
    }


def parse_json_object(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(raw[start : end + 1])


def openai_judge(
    client: OpenAI,
    model: str,
    query: str,
    expected_title: str,
    answer: str,
    retrieved_context: str,
) -> dict[str, Any]:
    prompt = {
        "query": query,
        "reference_title": expected_title,
        "rag_answer": answer,
        "retrieved_context": retrieved_context,
        "instructions": (
            "Score only from the provided query, retrieved_context, and rag_answer. "
            "The movies in retrieved_context are in retrieval rank order from top to bottom. "
            "This is a recommendation-quality evaluation, not a strict exact-title lookup. "
            "Exact-title retrieval is reported separately; judge recommendation quality from "
            "retrieved_context and rag_answer. A good answer recommends movies that are present "
            "in retrieved_context and match the query, or says the context is insufficient when appropriate. "
            "Return strict JSON with keys accuracy, faithfulness, answer_relevance, "
            "context_relevance, semantic_context_precision, semantic_context_recall, "
            "semantic_retrieval_hit, semantic_retrieval_top1, semantic_retrieval_rank, "
            "semantic_reciprocal_rank, "
            "hallucination_detected, uses_correct_context, "
            "expected_title_mentioned, explanation. Scores are integers from 1 to 5. "
            "accuracy means the recommendations satisfy the query, not whether they match reference_title. "
            "semantic_context_precision means how much of retrieved_context is relevant to the query. "
            "semantic_context_recall means how completely retrieved_context covers the query intent and constraints. "
            "semantic_retrieval_hit means at least one retrieved movie is semantically relevant to the query. "
            "semantic_retrieval_top1 means the first retrieved movie is semantically relevant to the query. "
            "semantic_retrieval_rank is the 1-based rank of the first semantically relevant retrieved movie, or null if none. "
            "semantic_reciprocal_rank is 1 divided by semantic_retrieval_rank, or 0 if none. "
            "uses_correct_context means the answer uses movies and facts from retrieved_context. "
            "hallucination_detected, uses_correct_context, and expected_title_mentioned "
            "are booleans. explanation is one concise sentence."
        ),
    }
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a strict evaluator for a movie RAG system.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    )
    content = response.choices[0].message.content or "{}"
    parsed = parse_json_object(content)

    return {
        "accuracy": clamp_score(parsed.get("accuracy")),
        "faithfulness": clamp_score(parsed.get("faithfulness")),
        "answer_relevance": clamp_score(parsed.get("answer_relevance")),
        "context_relevance": clamp_score(parsed.get("context_relevance")),
        "semantic_context_precision": clamp_score(
            parsed.get("semantic_context_precision", parsed.get("context_relevance"))
        ),
        "semantic_context_recall": clamp_score(
            parsed.get("semantic_context_recall", parsed.get("context_relevance"))
        ),
        "semantic_retrieval_hit": bool(parsed.get("semantic_retrieval_hit", False)),
        "semantic_retrieval_top1": bool(parsed.get("semantic_retrieval_top1", False)),
        "semantic_retrieval_rank": optional_positive_int(
            parsed.get("semantic_retrieval_rank")
        ),
        "semantic_reciprocal_rank": reciprocal_rank_from_judge(parsed),
        "hallucination_detected": bool(parsed.get("hallucination_detected", False)),
        "uses_correct_context": bool(parsed.get("uses_correct_context", False)),
        "expected_title_mentioned": bool(parsed.get("expected_title_mentioned", False)),
        "explanation": str(parsed.get("explanation", "")),
    }


def normalize_judge_result(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": clamp_score(parsed.get("accuracy")),
        "faithfulness": clamp_score(parsed.get("faithfulness")),
        "answer_relevance": clamp_score(parsed.get("answer_relevance")),
        "context_relevance": clamp_score(parsed.get("context_relevance")),
        "semantic_context_precision": clamp_score(
            parsed.get("semantic_context_precision", parsed.get("context_relevance"))
        ),
        "semantic_context_recall": clamp_score(
            parsed.get("semantic_context_recall", parsed.get("context_relevance"))
        ),
        "semantic_retrieval_hit": bool(parsed.get("semantic_retrieval_hit", False)),
        "semantic_retrieval_top1": bool(parsed.get("semantic_retrieval_top1", False)),
        "semantic_retrieval_rank": optional_positive_int(
            parsed.get("semantic_retrieval_rank")
        ),
        "semantic_reciprocal_rank": reciprocal_rank_from_judge(parsed),
        "hallucination_detected": bool(parsed.get("hallucination_detected", False)),
        "uses_correct_context": bool(parsed.get("uses_correct_context", False)),
        "expected_title_mentioned": bool(parsed.get("expected_title_mentioned", False)),
        "explanation": str(parsed.get("explanation", "")),
    }


def openai_judge_batch(
    client: OpenAI,
    model: str,
    cases: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    prompt = {
        "cases": cases,
        "instructions": (
            "Evaluate every case independently. Score only from each case's query, "
            "retrieved_context, and rag_answer. The movies in retrieved_context are in "
            "retrieval rank order from top to bottom. This is a recommendation-quality evaluation, "
            "not a strict exact-title lookup. Exact-title retrieval is reported separately; "
            "judge recommendation quality from retrieved_context and rag_answer. A good answer "
            "recommends movies that are present in retrieved_context and match the query, or says "
            "the context is insufficient when appropriate. Return strict JSON with one key: results. "
            "results must be an array with one object per input case. Each object must include "
            "case_id, accuracy, faithfulness, answer_relevance, context_relevance, "
            "semantic_context_precision, semantic_context_recall, "
            "semantic_retrieval_hit, semantic_retrieval_top1, semantic_retrieval_rank, "
            "semantic_reciprocal_rank, "
            "hallucination_detected, uses_correct_context, expected_title_mentioned, explanation. "
            "Scores are integers from 1 to 5. accuracy means the recommendations satisfy the query, "
            "not whether they match reference_title. uses_correct_context means the answer "
            "uses movies and facts from retrieved_context. semantic_context_precision means how much "
            "of retrieved_context is relevant to the query. semantic_context_recall means how completely "
            "retrieved_context covers the query intent and constraints. semantic_retrieval_hit means "
            "at least one retrieved movie is semantically relevant to the query. semantic_retrieval_top1 "
            "means the first retrieved movie is semantically relevant to the query. semantic_retrieval_rank "
            "is the 1-based rank of the first semantically relevant retrieved movie, or null if none. "
            "semantic_reciprocal_rank is 1 divided by semantic_retrieval_rank, or 0 if none. "
            "Booleans must be true or false. "
            "explanation must be one concise sentence."
        ),
    }
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a strict evaluator for a movie RAG system.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    )
    content = response.choices[0].message.content or "{}"
    parsed = parse_json_object(content)
    raw_results = parsed.get("results", [])
    if not isinstance(raw_results, list):
        raise ValueError("Batch judge response missing results array.")

    results: dict[str, dict[str, Any]] = {}
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        case_id = str(item.get("case_id", "")).strip()
        if case_id:
            results[case_id] = normalize_judge_result(item)

    missing = [str(case["case_id"]) for case in cases if str(case["case_id"]) not in results]
    if missing:
        raise ValueError(f"Batch judge response missing case ids: {missing}")

    return results


def mean(values: list[float]) -> float | None:
    return None if not values else statistics.fmean(values)


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def num(value: float | int | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def case_rag_score(judge: dict[str, Any]) -> float | None:
    required_judge_metrics = [
        "accuracy",
        "faithfulness",
        "answer_relevance",
        "context_relevance",
        "semantic_context_precision",
        "semantic_context_recall",
    ]
    if any(metric not in judge for metric in required_judge_metrics):
        return None

    components = [
        score_ratio(judge.get("accuracy")),
        score_ratio(judge.get("faithfulness")),
        score_ratio(judge.get("answer_relevance")),
        score_ratio(judge.get("context_relevance")),
        score_ratio(judge.get("semantic_context_precision")),
        score_ratio(judge.get("semantic_context_recall")),
    ]
    valid = [float(value) for value in components if value is not None]
    return mean(valid)


def case_passed(
    rag_score: float | None,
    judge: dict[str, Any],
    wrong_context: bool,
    threshold: float,
) -> bool:
    if rag_score is None:
        return False
    return bool(
        rag_score >= threshold
        and not judge.get("hallucination_detected", False)
        and not wrong_context
    )


def render_dashboard_image(output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    fig = plt.figure(figsize=(16, 10), facecolor="#f6f7f9")
    grid = fig.add_gridspec(3, 3, height_ratios=[0.8, 1.1, 1.1], hspace=0.42, wspace=0.28)
    fig.suptitle(
        f"RAG Evaluation Dashboard | {summary['run_id']} | Judge: {summary['openai_judge_model']}",
        fontsize=18,
        fontweight="bold",
        x=0.03,
        ha="left",
    )

    metrics = [
        ("Cases", str(summary["case_count"])),
        ("RAG Score", pct(summary["rag_score"])),
        ("Pass / Fail", f"{summary['pass_count']} / {summary['fail_count']}"),
        ("Hallucination", pct(summary["hallucination_rate"])),
        ("Wrong Context", pct(summary["wrong_context_rate"])),
        ("Avg Distance", num(summary["mean_avg_distance"])),
    ]
    card_grid = grid[0, :].subgridspec(2, 3, hspace=0.18, wspace=0.22)
    for idx, (label, value) in enumerate(metrics):
        ax = fig.add_subplot(card_grid[idx // 3, idx % 3])
        ax.set_facecolor("white")
        ax.text(0.05, 0.67, label, fontsize=12, color="#52606d", transform=ax.transAxes)
        ax.text(0.05, 0.22, value, fontsize=25, fontweight="bold", color="#1f2933", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d9e2ec")

    ax_core = fig.add_subplot(grid[1, :2])
    core_labels = [
        "Accuracy",
        "Faithfulness",
        "Relevance",
        "Retrieval Precision",
        "Retrieval Recall",
    ]
    core_values = [
        summary["mean_accuracy_ratio"] or 0,
        summary["mean_faithfulness_ratio"] or 0,
        summary["mean_answer_relevance_ratio"] or 0,
        summary["mean_context_precision"] or 0,
        summary["mean_context_recall"] or 0,
    ]
    ax_core.bar(core_labels, core_values, color=["#2f80ed", "#0b6b3a", "#9b51e0", "#f2994a", "#56ccf2"])
    ax_core.set_ylim(0, 1)
    ax_core.set_title("Core RAG Metrics")
    ax_core.tick_params(axis="x", rotation=18)
    ax_core.grid(axis="y", alpha=0.25)

    ax_pass = fig.add_subplot(grid[1, 2])
    ax_pass.pie(
        [summary["pass_count"], summary["fail_count"]],
        labels=["Pass", "Fail"],
        autopct=lambda value: "" if value == 0 else f"{value:.0f}%",
        colors=["#0b6b3a", "#b42318"],
        startangle=90,
    )
    ax_pass.set_title("Pass / Fail")

    ax_risk = fig.add_subplot(grid[2, 0])
    ax_risk.bar(
        ["Hallucination", "Wrong Context"],
        [summary["hallucination_rate"] or 0, summary["wrong_context_rate"] or 0],
        color=["#b42318", "#f2994a"],
    )
    ax_risk.set_ylim(0, 1)
    ax_risk.set_title("Risk Rates")
    ax_risk.grid(axis="y", alpha=0.25)

    ax_latency = fig.add_subplot(grid[2, 1])
    latencies = [float(row.get("latency_seconds") or 0) for row in rows]
    ax_latency.plot(range(1, len(latencies) + 1), latencies, marker="o", color="#2f80ed")
    ax_latency.set_title("Latency Per Case")
    ax_latency.set_xlabel("Case")
    ax_latency.set_ylabel("Seconds")
    ax_latency.grid(alpha=0.25)

    ax_retrieval = fig.add_subplot(grid[2, 2])
    ax_retrieval.bar(
        ["Sem. Precision", "Sem. Recall", "Ctx Rel."],
        [
            summary["mean_context_precision"] or 0,
            summary["mean_context_recall"] or 0,
            summary["mean_context_relevance_ratio"] or 0,
        ],
        color=["#2f80ed", "#0b6b3a", "#56ccf2"],
    )
    ax_retrieval.set_ylim(0, 1)
    ax_retrieval.set_title(
        f"Retrieval | Top Rerank: {num(summary['mean_top_rerank_score'])}"
    )
    ax_retrieval.grid(axis="y", alpha=0.25)

    fig.savefig(output_dir / "dashboard.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_metric_bars_image(output_dir: Path, summary: dict[str, Any]) -> None:
    metrics = [
        ("Faithfulness", summary["mean_faithfulness_ratio"]),
        ("Answer relevance", summary["mean_answer_relevance_ratio"]),
        ("Context relevance", summary["mean_context_relevance_ratio"]),
        ("Semantic precision", summary["mean_context_precision"]),
        ("Semantic recall", summary["mean_context_recall"]),
        ("RAG score", summary["rag_score"]),
    ]
    labels = [label for label, _value in metrics]
    values = [0 if value is None else float(value) for _label, value in metrics]
    weakest_index = min(range(len(values)), key=lambda idx: values[idx])
    colors = ["#2f80ed"] * len(values)
    colors[weakest_index] = "#b42318"

    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#f6f7f9")
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_title("Metric Comparison", loc="left", fontsize=16, fontweight="bold")
    ax.set_ylabel("Score ratio")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, min(0.98, value + 0.03), pct(value), ha="center", fontsize=10)
    ax.text(
        weakest_index,
        max(0.08, values[weakest_index] / 2),
        "Weakest",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )

    fig.savefig(output_dir / "metric_bars.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_cases_image(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    display_rows = []
    for row in rows[:30]:
        retrieval = row.get("retrieval", {})
        judge = row.get("judge", {})
        top_titles = ", ".join(str(title) for title in retrieval.get("retrieved_titles", [])[:2])
        display_rows.append(
            [
                row["case_id"],
                str(row["expected_title"])[:28],
                "pass" if row.get("passed") else "fail",
                pct(row.get("rag_score")),
                pct(score_ratio(judge.get("semantic_context_precision"))),
                pct(score_ratio(judge.get("semantic_context_recall"))),
                str(judge.get("faithfulness", "N/A")),
                str(judge.get("answer_relevance", "N/A")),
                "Y" if judge.get("hallucination_detected") else "N",
                top_titles[:38],
            ]
        )

    height = max(4.5, 1.0 + 0.34 * max(1, len(display_rows)))
    fig, ax = plt.subplots(figsize=(16, height), facecolor="#f6f7f9")
    ax.axis("off")
    ax.set_title("Per-Case Evaluation Results", loc="left", fontsize=16, fontweight="bold", pad=14)

    table = ax.table(
        cellText=display_rows,
        colLabels=[
            "Case",
            "Expected",
            "Status",
            "RAG Score",
            "Sem Prec.",
            "Sem Rec.",
            "Faith.",
            "Ans Rel.",
            "Halluc.",
            "Top Retrieved",
        ],
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        colWidths=[0.07, 0.16, 0.07, 0.09, 0.09, 0.09, 0.07, 0.08, 0.07, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e2ec")
        if row_idx == 0:
            cell.set_facecolor("#eef2f6")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")

    fig.savefig(output_dir / "cases.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_outputs(output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "use_case",
                "expected_title",
                "search_query",
                "metadata_filter_applied",
                "rag_score",
                "passed",
                "accuracy",
                "rank",
                "avg_distance",
                "best_distance",
                "top_distance",
                "avg_rerank_score",
                "top_rerank_score",
                "best_rerank_score",
                "faithfulness",
                "answer_relevance",
                "context_relevance",
                "semantic_context_precision",
                "semantic_context_recall",
                "semantic_retrieval_hit",
                "semantic_retrieval_top1",
                "semantic_retrieval_rank",
                "semantic_reciprocal_rank",
                "context_precision",
                "context_recall",
                "exact_title_precision",
                "exact_title_recall",
                "exact_match",
                "reference_title_retrieved",
                "hallucination_detected",
                "uses_correct_context",
                "wrong_context",
                "expected_title_mentioned",
                "generation_error",
                "judge_error",
            ],
        )
        writer.writeheader()
        for row in rows:
            retrieval = row.get("retrieval", {})
            judge = row.get("judge", {})
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "use_case": row.get("use_case"),
                    "expected_title": row["expected_title"],
                    "search_query": row.get("search_query"),
                    "metadata_filter_applied": row.get("metadata_filter_applied"),
                    "rag_score": row.get("rag_score"),
                    "passed": row.get("passed"),
                    "accuracy": judge.get("accuracy"),
                    "rank": retrieval.get("rank"),
                    "avg_distance": retrieval.get("avg_distance"),
                    "best_distance": retrieval.get("best_distance"),
                    "top_distance": retrieval.get("top_distance"),
                    "avg_rerank_score": retrieval.get("avg_rerank_score"),
                    "top_rerank_score": retrieval.get("top_rerank_score"),
                    "best_rerank_score": retrieval.get("best_rerank_score"),
                    "faithfulness": judge.get("faithfulness"),
                    "answer_relevance": judge.get("answer_relevance"),
                    "context_relevance": judge.get("context_relevance"),
                    "semantic_context_precision": judge.get("semantic_context_precision"),
                    "semantic_context_recall": judge.get("semantic_context_recall"),
                    "semantic_retrieval_hit": judge.get("semantic_retrieval_hit"),
                    "semantic_retrieval_top1": judge.get("semantic_retrieval_top1"),
                    "semantic_retrieval_rank": judge.get("semantic_retrieval_rank"),
                    "semantic_reciprocal_rank": judge.get("semantic_reciprocal_rank"),
                    "context_precision": score_ratio(judge.get("semantic_context_precision")),
                    "context_recall": score_ratio(judge.get("semantic_context_recall")),
                    "exact_title_precision": retrieval.get("exact_title_precision"),
                    "exact_title_recall": retrieval.get("exact_title_recall"),
                    "exact_match": row.get("exact_match"),
                    "reference_title_retrieved": row.get("reference_title_retrieved"),
                    "hallucination_detected": judge.get("hallucination_detected"),
                    "uses_correct_context": judge.get("uses_correct_context"),
                    "wrong_context": row.get("wrong_context"),
                    "expected_title_mentioned": judge.get("expected_title_mentioned"),
                    "generation_error": row.get("generation_error"),
                    "judge_error": row.get("judge_error"),
                }
            )

    render_dashboard_image(output_dir, summary, rows)
    render_metric_bars_image(output_dir, summary)
    render_cases_image(output_dir, rows)


def run(args: argparse.Namespace) -> Path:
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in environment or .env")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = RESULTS_DIR / run_id
    dataset_path = ROOT_DIR / args.dataset
    df = load_filtered_dataset(dataset_path, args.min_rating)
    train_df, eval_case_df = make_holdout_split(df, args.sample_size, args.seed)
    indexed_corpus_df = df.reset_index(drop=True).copy()
    indexed_corpus_df.insert(0, "source_row_id", range(len(indexed_corpus_df)))
    evaluation_mode = "holdout_eval_full_vector_index"
    cases = make_eval_cases(
        eval_df=eval_case_df,
        sample_size=args.sample_size,
        seed=args.seed,
        split_label="holdout_eval_full_index",
    )
    train_titles = set(train_df["Title"].map(normalized_title))

    config = RAGConfig.from_env()
    train_chunks_path = output_dir / "full_movie_chunks_metadata.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    eval_case_df.to_csv(output_dir / "eval_case_source.csv", index=False)
    indexed_corpus_df.to_csv(output_dir / "indexed_corpus.csv", index=False)
    build_train_chunks(indexed_corpus_df, train_chunks_path)
    relevant_chunk_counts = load_relevant_chunk_counts(train_chunks_path)
    eval_chroma_path = output_dir / "chroma_full"
    store = MovieChromaStore(
        persist_path=str(eval_chroma_path),
        collection_name=config.chroma_collection,
        embedding_model_name=config.embedding_model_name,
    )
    store.build_from_chunks_csv(train_chunks_path, reset=True)
    pipeline = MovieRAGPipeline(store, config)
    client = OpenAI(api_key=openai_key)

    rows: list[dict[str, Any]] = []
    pending_judge: list[dict[str, Any]] = []

    for case in cases:
        started = time.perf_counter()
        generation_error = None
        try:
            rag_response = pipeline.query(case.query, top_k=args.top_k)
        except Exception as exc:
            generation_error = str(exc)
            results = store.search(case.query, top_k=args.top_k)
            rag_response = {"answer": "", "results": results}

        results = rag_response.get("results", [])
        retrieval = retrieval_metrics(results, case.expected_title, relevant_chunk_counts)
        context = MovieRAGPipeline.format_context(results)

        answer = rag_response.get("answer", "")
        exact_match = normalized_title(case.expected_title) in normalized_title(answer)
        reference_title_retrieved = bool(retrieval.get("reference_title_retrieved"))
        judge_error = None if answer else "No RAG answer generated."

        row = {
            **asdict(case),
            "retrieval": retrieval,
            "answer": answer,
            "structured_query": rag_response.get("structured_query"),
            "metadata_filter": rag_response.get("metadata_filter"),
            "metadata_filter_applied": rag_response.get("metadata_filter_applied"),
            "search_query": rag_response.get("search_query"),
            "judge": {},
            "exact_match": exact_match,
            "reference_title_retrieved": reference_title_retrieved,
            "rag_score": None,
            "wrong_context": True,
            "passed": False,
            "generation_error": generation_error,
            "judge_error": judge_error,
            "latency_seconds": time.perf_counter() - started,
        }
        rows.append(row)

        if answer:
            pending_judge.append(
                {
                    "case_id": case.case_id,
                    "query": case.query,
                    "reference_title": case.expected_title,
                    "rag_answer": answer,
                    "retrieved_context": context,
                }
            )

    rows_by_case_id = {row["case_id"]: row for row in rows}
    for start in range(0, len(pending_judge), args.judge_batch_size):
        batch = pending_judge[start : start + args.judge_batch_size]
        try:
            batch_results = openai_judge_batch(
                client=client,
                model=args.openai_judge_model,
                cases=batch,
            )
            judge_error = None
        except Exception as exc:
            batch_results = {}
            judge_error = str(exc)

        for case_payload in batch:
            row = rows_by_case_id[case_payload["case_id"]]
            if judge_error:
                row["judge_error"] = judge_error
                continue
            judge = batch_results[case_payload["case_id"]]
            rag_score = case_rag_score(judge)
            wrong_context = bool(not judge.get("uses_correct_context", False))
            row["judge"] = judge
            row["rag_score"] = rag_score
            row["wrong_context"] = wrong_context
            row["passed"] = case_passed(
                rag_score,
                judge,
                wrong_context,
                args.pass_threshold,
            )

    judge_values: dict[str, list[float]] = {
        "accuracy": [],
        "faithfulness": [],
        "answer_relevance": [],
        "context_relevance": [],
        "semantic_context_precision": [],
        "semantic_context_recall": [],
    }
    context_precision_values: list[float] = []
    context_recall_values: list[float] = []
    rag_score_values: list[float] = []
    exact_match_values: list[float] = []
    metadata_filter_applied_values: list[float] = []
    hallucination_values: list[float] = []
    wrong_context_values: list[float] = []
    semantic_hit_values: list[float] = []
    semantic_top1_values: list[float] = []
    semantic_reciprocal_rank_values: list[float] = []
    exact_hit_values: list[float] = []
    exact_top1_values: list[float] = []
    exact_reciprocal_rank_values: list[float] = []
    avg_distance_values: list[float] = []
    best_distance_values: list[float] = []
    top_distance_values: list[float] = []
    avg_rerank_score_values: list[float] = []
    top_rerank_score_values: list[float] = []
    best_rerank_score_values: list[float] = []

    for row in rows:
        if row.get("rag_score") is not None:
            rag_score_values.append(float(row["rag_score"]))
        exact_match_values.append(1.0 if row.get("exact_match") else 0.0)
        metadata_filter_applied_values.append(
            1.0 if row.get("metadata_filter_applied") else 0.0
        )
        retrieval = row["retrieval"]
        exact_hit_values.append(1.0 if retrieval.get("exact_title_hit") else 0.0)
        exact_top1_values.append(1.0 if retrieval.get("exact_title_top1_match") else 0.0)
        exact_reciprocal_rank_values.append(
            float(retrieval.get("exact_title_reciprocal_rank") or 0.0)
        )
        if "semantic_retrieval_hit" in row["judge"]:
            semantic_hit_values.append(
                1.0 if row["judge"].get("semantic_retrieval_hit") else 0.0
            )
        if "semantic_retrieval_top1" in row["judge"]:
            semantic_top1_values.append(
                1.0 if row["judge"].get("semantic_retrieval_top1") else 0.0
            )
        if "semantic_reciprocal_rank" in row["judge"]:
            semantic_reciprocal_rank_values.append(
                float(row["judge"].get("semantic_reciprocal_rank") or 0.0)
            )
        if retrieval.get("avg_distance") is not None:
            avg_distance_values.append(float(retrieval["avg_distance"]))
        if retrieval.get("best_distance") is not None:
            best_distance_values.append(float(retrieval["best_distance"]))
        if retrieval.get("top_distance") is not None:
            top_distance_values.append(float(retrieval["top_distance"]))
        if retrieval.get("avg_rerank_score") is not None:
            avg_rerank_score_values.append(float(retrieval["avg_rerank_score"]))
        if retrieval.get("top_rerank_score") is not None:
            top_rerank_score_values.append(float(retrieval["top_rerank_score"]))
        if retrieval.get("best_rerank_score") is not None:
            best_rerank_score_values.append(float(retrieval["best_rerank_score"]))
        semantic_precision = score_ratio(row["judge"].get("semantic_context_precision"))
        if semantic_precision is not None:
            context_precision_values.append(float(semantic_precision))
        semantic_recall = score_ratio(row["judge"].get("semantic_context_recall"))
        if semantic_recall is not None:
            context_recall_values.append(float(semantic_recall))
        if "hallucination_detected" in row["judge"]:
            hallucination_values.append(
                1.0 if row["judge"].get("hallucination_detected") else 0.0
            )
        wrong_context_values.append(1.0 if row.get("wrong_context") else 0.0)
        for metric in judge_values:
            if metric in row["judge"]:
                judge_values[metric].append(float(row["judge"][metric]))

    judge_means = {f"mean_{metric}": mean(values) for metric, values in judge_values.items()}
    judge_ratio_means = {
        f"{metric}_ratio": score_ratio(value)
        for metric, value in judge_means.items()
    }
    pass_count = sum(1 for row in rows if row.get("passed"))
    fail_count = len(rows) - pass_count
    train_source_ids = set(train_df["source_row_id"])
    eval_source_ids = set(eval_case_df["source_row_id"])
    train_eval_source_overlap_count = len(train_source_ids.intersection(eval_source_ids))

    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "case_count": len(rows),
        "dataset": str(dataset_path),
        "evaluation_mode": evaluation_mode,
        "reference_title_excluded_from_index": False,
        "dataset_row_count_after_filter": int(len(df)),
        "train_row_count": int(len(train_df)),
        "eval_case_source_row_count": int(len(eval_case_df)),
        "indexed_corpus_row_count": int(len(indexed_corpus_df)),
        "train_eval_source_overlap_count": train_eval_source_overlap_count,
        "eval_case_source_excluded_from_train": train_eval_source_overlap_count == 0,
        "train_unique_title_count": len(train_titles),
        "indexed_corpus_unique_title_count": len(
            set(indexed_corpus_df["Title"].map(normalized_title))
        ),
        "use_case_counts": {
            use_case: sum(1 for row in rows if row.get("use_case") == use_case)
            for use_case in sorted({row.get("use_case") for row in rows})
        },
        "train_chunks_path": str(train_chunks_path),
        "train_split_path": str(output_dir / "train_split.csv"),
        "eval_case_source_path": str(output_dir / "eval_case_source.csv"),
        "indexed_corpus_path": str(output_dir / "indexed_corpus.csv"),
        "eval_chroma_path": str(eval_chroma_path),
        "top_k": args.top_k,
        "seed": args.seed,
        "min_rating": args.min_rating,
        "openai_judge_model": args.openai_judge_model,
        "judge_batch_size": args.judge_batch_size,
        "rag_llm_model": config.llm_model_name,
        "embedding_model": config.embedding_model_name,
        "reranker_model": config.reranker_model_name,
        "rerank_fetch_k": config.rerank_fetch_k if config.enable_reranking else None,
        "enable_reranking": config.enable_reranking,
        "enable_query_structuring": config.enable_query_structuring,
        "enable_llm_query_structuring": config.enable_llm_query_structuring,
        "chroma_collection": config.chroma_collection,
        "pass_threshold": args.pass_threshold,
        "rag_score": mean(rag_score_values),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": None if not rows else pass_count / len(rows),
        "fail_rate": None if not rows else fail_count / len(rows),
        "hallucination_rate": mean(hallucination_values),
        "wrong_context_rate": mean(wrong_context_values),
        "exact_match_rate": mean(exact_match_values),
        "metadata_filter_applied_rate": mean(metadata_filter_applied_values),
        "exact_title_hit_rate": mean(exact_hit_values),
        "exact_title_top1_rate": mean(exact_top1_values),
        "mean_exact_title_reciprocal_rank": mean(exact_reciprocal_rank_values),
        "retrieval_hit_rate": mean(semantic_hit_values),
        "retrieval_top1_rate": mean(semantic_top1_values),
        "mean_reciprocal_rank": mean(semantic_reciprocal_rank_values),
        "mean_avg_distance": mean(avg_distance_values),
        "mean_best_distance": mean(best_distance_values),
        "mean_top_distance": mean(top_distance_values),
        "mean_avg_rerank_score": mean(avg_rerank_score_values),
        "mean_top_rerank_score": mean(top_rerank_score_values),
        "mean_best_rerank_score": mean(best_rerank_score_values),
        "mean_context_precision": mean(context_precision_values),
        "mean_context_recall": mean(context_recall_values),
        **judge_means,
        **judge_ratio_means,
        "generation_error_count": sum(1 for row in rows if row.get("generation_error")),
        "judge_error_count": sum(1 for row in rows if row.get("judge_error")),
    }

    write_outputs(output_dir, summary, rows)
    return output_dir


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate the movie RAG pipeline.")
    parser.add_argument(
        "--dataset",
        default="data/processed/IMDb_Dataset_Composite_Cleaned.csv",
        help="CSV used to create grounded evaluation cases.",
    )
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-rating", type=float, default=None)
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=0.7,
        help="Minimum per-case RAG score ratio required for pass.",
    )
    parser.add_argument(
        "--openai-judge-model",
        default=os.getenv("OPENAI_EVAL_MODEL", "gpt-5.5"),
    )
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=20,
        help="Number of cases scored in one OpenAI judge request.",
    )
    args = parser.parse_args()
    if args.judge_batch_size < 1:
        raise ValueError("--judge-batch-size must be at least 1")

    output_dir = run(args)
    print(f"Evaluation results saved to: {output_dir}")


if __name__ == "__main__":
    main()
