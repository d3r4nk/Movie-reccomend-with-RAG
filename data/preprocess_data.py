#Preprocess IMDb data using the same logic as the RAG-MOVIE-REC repo.

#Outputs:
#- IMDb_Dataset_Composite.csv
#- IMDb_Dataset_Composite_Cleaned.csv
#- Movie_Descriptions.csv
#- movie_chunks_metadata.csv


from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


RAW_FILENAMES = ("IMDb_Dataset.csv", "IMDb_Dataset_2.csv", "IMDb_Dataset_3.csv")
CLEANED_COLUMNS = [
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
]


def combine_genres(row: pd.Series) -> str:
    genres = [row.get("Genre"), row.get("Second_Genre"), row.get("Third_Genre")]
    genres = [g for g in genres if pd.notnull(g) and str(g).strip() != ""]
    return ", ".join(str(g).strip() for g in genres)


def add_delimiters(name_string: str) -> str:
    """Match the original repo's Star Cast delimiter cleanup."""
    if pd.isna(name_string):
        return name_string

    initial_names: dict[str, str] = {}

    def protect_initials(match: re.Match[str]) -> str:
        key = f"__INITIAL_{len(initial_names)}__"
        initial_names[key] = match.group(0)
        return key

    protected_string = re.sub(
        r"[A-Z]\.[A-Z]\.[A-Z]\.\s*\w+",
        protect_initials,
        str(name_string),
    )
    split_names = re.sub(r"([a-z])([A-Z])", r"\1,\2", protected_string)
    split_names = re.sub(r"\s+", ",", split_names.strip())
    split_names = re.sub(r"(\w+),(\w+)(,|$)", r"\1 \2\3", split_names)

    for key, value in initial_names.items():
        split_names = split_names.replace(key, value)

    return re.sub(r",+", ",", split_names)


def merge_raw_files(raw_dir: str | Path) -> pd.DataFrame:
    """Merge Kaggle CSVs using the original update-by-title behavior."""
    raw_path = Path(raw_dir)
    composite_df = pd.DataFrame()

    for filename in RAW_FILENAMES:
        file_path = raw_path / filename
        if not file_path.exists():
            print(f"Warning: {file_path} not found. Skipping.")
            continue

        df = pd.read_csv(file_path, encoding="utf-8")
        df = df.drop_duplicates(subset="Title", keep="last")

        if composite_df.empty:
            composite_df = df.copy()
            print(f"Merged file: {filename}")
            continue

        df = df.set_index("Title")
        composite_df = composite_df.set_index("Title")
        composite_df.update(df)
        composite_df = pd.concat(
            [composite_df, df[~df.index.isin(composite_df.index)]]
        ).reset_index()
        print(f"Merged file: {filename}")

    if composite_df.empty:
        raise FileNotFoundError(f"No raw CSV files found in {raw_path.resolve()}")

    return composite_df


def clean_composite(composite_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same missing-value, genre rollup, and cast cleanup rules."""
    df = composite_df.copy()

    df["Poster-src"] = df["Poster-src"].fillna("No Poster Available")
    df["Second_Genre"] = df["Second_Genre"].fillna("")
    df["Third_Genre"] = df["Third_Genre"].fillna("")

    df["Genres"] = df.apply(combine_genres, axis=1)
    df = df.drop(columns=["Genre", "Second_Genre", "Third_Genre"])
    df["Star Cast"] = df["Star Cast"].apply(add_delimiters)

    return df[CLEANED_COLUMNS]


def format_star_cast_for_description(star_cast: str) -> str:
    actors = [actor.strip() for actor in str(star_cast).split(",")]
    if len(actors) > 3:
        return ", ".join(actors[:3]) + ", and others"
    return ", ".join(actors)


def generate_description(row: pd.Series) -> str:
    """Generate the same description text shape as the original repo."""
    return (
        f"{row['Title']} ({int(row['Year'])}) is a {str(row['Genres']).lower()} "
        f"film directed by {row['Director']}. "
        f"Featuring {format_star_cast_for_description(row['Star Cast'])}, "
        f"this movie has an IMDb rating of {row['IMDb Rating']}/10 and a MetaScore "
        f"of {row['MetaScore']}. With a runtime of "
        f"{int(float(row['Duration (minutes)']))} minutes, it is rated "
        f"{row['Certificates']}."
    )


def create_descriptions(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    records = [
        {"Title": row["Title"], "Description": generate_description(row)}
        for _, row in cleaned_df.iterrows()
    ]
    return pd.DataFrame(records)


def chunk_descriptions(
    descriptions_df: pd.DataFrame,
    chunk_size: int = 200,
    chunk_overlap: int = 20,
) -> pd.DataFrame:
    """Chunk each movie description by character length, as in RAG-MOVIE-REC."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = []

    for _, row in descriptions_df.iterrows():
        for chunk in splitter.split_text(row["Description"]):
            chunks.append(
                {
                    "Title": row["Title"],
                    "Chunk": chunk,
                    "Metadata": {"Title": row["Title"]},
                }
            )

    return pd.DataFrame(chunks)


def preprocess(
    raw_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
    chunk_size: int = 200,
    chunk_overlap: int = 20,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    composite_df = merge_raw_files(raw_dir)
    composite_path = output_path / "IMDb_Dataset_Composite.csv"
    composite_df.to_csv(composite_path, index=False)

    cleaned_df = clean_composite(composite_df)
    cleaned_path = output_path / "IMDb_Dataset_Composite_Cleaned.csv"
    cleaned_df.to_csv(cleaned_path, index=False)

    descriptions_df = create_descriptions(cleaned_df)
    descriptions_path = output_path / "Movie_Descriptions.csv"
    descriptions_df.to_csv(descriptions_path, index=False)

    chunks_df = chunk_descriptions(descriptions_df, chunk_size, chunk_overlap)
    chunks_path = output_path / "movie_chunks_metadata.csv"
    chunks_df[["Title", "Chunk", "Metadata"]].to_csv(chunks_path, index=False)

    print(f"Cleaned movies: {len(cleaned_df)}")
    print(f"Descriptions: {len(descriptions_df)}")
    print(f"Chunks: {len(chunks_df)}")

    return {
        "composite": composite_path,
        "cleaned": cleaned_path,
        "descriptions": descriptions_path,
        "chunks": chunks_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess IMDb movie data.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory with raw CSVs.")
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for processed CSV outputs.",
    )
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--chunk-overlap", type=int, default=20)
    args = parser.parse_args()

    preprocess(args.raw_dir, args.output_dir, args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()
