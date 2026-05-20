from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from langchain_openai import ChatOpenAI


GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Biography",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "Film-Noir",
    "History",
    "Horror",
    "Music",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Sport",
    "Thriller",
    "War",
    "Western",
]

CERTIFICATES = ["G", "PG", "PG-13", "R", "NC-17", "TV-G", "TV-PG", "TV-14", "TV-MA"]


@dataclass
class StructuredMovieQuery:
    """Biểu diễn query người dùng thành ý nghĩa và filter.
    semantic_query dùng cho vector search. Các trường như genres, year_min,
    rating_min, actors dùng làm metadata filter trong ChromaDB.
    """
    semantic_query: str
    genres: list[str]
    actors: list[str] | None = None
    year_min: int | None = None
    year_max: int | None = None
    rating_min: float | None = None
    rating_max: float | None = None
    duration_min: int | None = None
    duration_max: int | None = None
    certificates: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Chuyển dataclass thành dict de log, debug, va tra ve qua API."""
        return asdict(self)


def _json_object(raw: str) -> dict[str, Any]:
    """Lấy object JSON từ text LLM trả về, ke ca khi text co phan thua."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(raw[start : end + 1])


def _number(value: Any) -> float | None:
    """Doi gia tri bat ky thanh float neu hop le, nguoc lai tra None."""
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    """Doi gia tri thanh int sau khi doc duoc dang so."""
    number = _number(value)
    return None if number is None else int(number)


def _clean_genres(values: Any) -> list[str]:
    """Giu lai chi cac genre hop le de filter khong bi sai ten cot."""
    if not isinstance(values, list):
        return []
    valid = {genre.casefold(): genre for genre in GENRES}
    cleaned = []
    for value in values:
        genre = valid.get(str(value).strip().casefold())
        if genre and genre not in cleaned:
            cleaned.append(genre)
    return cleaned


def _clean_certificates(values: Any) -> list[str]:
    """Giu lai chi cac certificate hop le nhu G, PG, PG-13, R."""
    if not isinstance(values, list):
        return []
    valid = {cert.casefold(): cert for cert in CERTIFICATES}
    cleaned = []
    for value in values:
        cert = valid.get(str(value).strip().casefold())
        if cert and cert not in cleaned:
            cleaned.append(cert)
    return cleaned


def _clean_actors(values: Any) -> list[str]:
    """Chuan hoa actor names do LLM/heuristic trich xuat de dung filter."""
    if not isinstance(values, list):
        return []
    cleaned = []
    for value in values:
        actor = " ".join(str(value).strip().split())
        if not actor:
            continue
        if actor.casefold() in {item.casefold() for item in cleaned}:
            continue
        cleaned.append(actor)
    return cleaned


class MovieQueryStructurer:
    """Tach query phim thanh semantic query va metadata constraints.

    Buoc nay giup RAG khong chi tim bang embedding, ma con loc theo nam,
    genre, rating, thoi luong, va certificate neu nguoi dung noi ro.
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        enable_llm: bool = True,
    ):
        """Nhan LLM tuy chon; neu tat LLM thi chi dung heuristic parser."""
        self.llm = llm
        self.enable_llm = enable_llm

    def structure(self, query: str) -> StructuredMovieQuery:
        """Chuyen query tu nhien thanh cau truc filter ket hop heuristic va LLM."""
        heuristic = self._heuristic(query)
        if not self.enable_llm or self.llm is None:
            return heuristic

        try:
            llm_structured = self._llm_structure(query)
        except Exception:
            return heuristic

        return self._merge(heuristic, llm_structured)

    def to_chroma_where(self, structured: StructuredMovieQuery) -> dict[str, Any] | None:
        """Biến StructuredMovieQuery thành cú pháp where của ChromaDB."""
        conditions: list[dict[str, Any]] = []

        for genre in structured.genres:
            conditions.append(
                {
                    "$or": [
                        {"Genre 1": {"$eq": genre}},
                        {"Genre 2": {"$eq": genre}},
                        {"Genre 3": {"$eq": genre}},
                    ]
                }
            )

        for actor in structured.actors or []:
            conditions.append(
                {
                    "$or": [
                        {"Actor 1": {"$eq": actor}},
                        {"Actor 2": {"$eq": actor}},
                        {"Actor 3": {"$eq": actor}},
                        {"Actor 4": {"$eq": actor}},
                        {"Actor 5": {"$eq": actor}},
                    ]
                }
            )

        if structured.year_min is not None:
            conditions.append({"Year": {"$gte": structured.year_min}})
        if structured.year_max is not None:
            conditions.append({"Year": {"$lte": structured.year_max}})
        if structured.rating_min is not None:
            conditions.append({"IMDb Rating": {"$gte": structured.rating_min}})
        if structured.rating_max is not None:
            conditions.append({"IMDb Rating": {"$lte": structured.rating_max}})
        if structured.duration_min is not None:
            conditions.append({"Duration (minutes)": {"$gte": structured.duration_min}})
        if structured.duration_max is not None:
            conditions.append({"Duration (minutes)": {"$lte": structured.duration_max}})
        if structured.certificates:
            if len(structured.certificates) == 1:
                conditions.append({"Certificates": {"$eq": structured.certificates[0]}})
            else:
                conditions.append(
                    {
                        "$or": [
                            {"Certificates": {"$eq": certificate}}
                            for certificate in structured.certificates
                        ]
                    }
                )

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _llm_structure(self, query: str) -> StructuredMovieQuery:
        """Hoi LLM de trich xuat constraints ro rang tu query nguoi dung."""
        prompt = {
            "task": "Extract only explicit movie search constraints from an English user query.",
            "rules": [
                "Return strict JSON only.",
                "Do not infer constraints that are not stated.",
                "Use null for unknown numeric bounds.",
                "Use only genres from valid_genres.",
                "Use only certificates from valid_certificates.",
                "Put explicitly named actors or cast members in actors.",
                "Do not put directors, characters, or vague cast-profile descriptions in actors.",
                "Preserve actor name capitalization from the query.",
                "semantic_query should remove numeric/filter-only constraints but keep the user's intent.",
            ],
            "valid_genres": GENRES,
            "valid_certificates": CERTIFICATES,
            "schema": {
                "semantic_query": "string",
                "genres": ["string"],
                "actors": ["string"],
                "year_min": "integer|null",
                "year_max": "integer|null",
                "rating_min": "number|null",
                "rating_max": "number|null",
                "duration_min": "integer|null",
                "duration_max": "integer|null",
                "certificates": ["string"],
            },
            "query": query,
        }
        response = self.llm.invoke(json.dumps(prompt, ensure_ascii=False))
        parsed = _json_object(str(response.content))
        return self._from_dict(query, parsed)

    def _from_dict(self, query: str, data: dict[str, Any]) -> StructuredMovieQuery:
        """Chuan hoa dict LLM tra ve thanh StructuredMovieQuery an toan."""
        semantic_query = str(data.get("semantic_query") or query).strip() or query
        return StructuredMovieQuery(
            semantic_query=semantic_query,
            genres=_clean_genres(data.get("genres")),
            actors=_clean_actors(data.get("actors")),
            year_min=_int(data.get("year_min")),
            year_max=_int(data.get("year_max")),
            rating_min=_number(data.get("rating_min")),
            rating_max=_number(data.get("rating_max")),
            duration_min=_int(data.get("duration_min")),
            duration_max=_int(data.get("duration_max")),
            certificates=_clean_certificates(data.get("certificates")),
        )

    def _heuristic(self, query: str) -> StructuredMovieQuery:
        """Dung regex de bat genre, nam, rating, runtime, certificate khong can LLM."""
        lowered = query.casefold()
        genres = [genre for genre in GENRES if re.search(rf"\b{re.escape(genre.casefold())}\b", lowered)]
        actors = self._heuristic_actors(query)
        certificates = []
        for cert in sorted(CERTIFICATES, key=len, reverse=True):
            if re.search(rf"(?<![\w-]){re.escape(cert.casefold())}(?![\w-])", lowered):
                certificates.append(cert)

        years = [int(match) for match in re.findall(r"\b(19\d{2}|20\d{2})\b", query)]
        year_min = None
        year_max = None
        if years:
            year = years[0]
            if re.search(r"\b(around|about|circa|near)\b", lowered):
                year_min = year - 5
                year_max = year + 5
            else:
                year_min = year
                year_max = year

        rating_min = None
        rating_max = None
        rating_match = re.search(r"\b(\d(?:\.\d+)?)\s*/\s*10\b", lowered)
        if not rating_match:
            rating_match = re.search(r"(?:rating|imdb)[^\d]*(\d(?:\.\d+)?)", lowered)
        if rating_match:
            rating_value = float(rating_match.group(1))
            if re.search(r"\b(above|over|at least|minimum|min|higher than|highly rated|high rated)\b", lowered):
                rating_min = rating_value
            elif re.search(r"\b(under|below|less than|maximum|max)\b", lowered):
                rating_max = rating_value
            else:
                rating_min = max(0.0, rating_value - 0.5)
                rating_max = min(10.0, rating_value + 0.5)
        elif re.search(r"\b(highly rated|high rated|strong imdb|good rating)\b", lowered):
            rating_min = 7.0

        duration_min = None
        duration_max = None
        duration_match = re.search(r"(\d{2,3})\s*(?:minutes|min|mins)", lowered)
        if duration_match:
            minutes = int(duration_match.group(1))
            if re.search(r"\b(under|below|less than|shorter than|max|maximum)\b", lowered):
                duration_max = minutes
            elif re.search(r"\b(over|above|longer than|at least|min|minimum)\b", lowered):
                duration_min = minutes
            else:
                duration_min = max(0, minutes - 10)
                duration_max = minutes + 10
        elif re.search(r"\b(under|less than|below)\s+2\s+hours\b", lowered):
            duration_max = 120
        elif re.search(r"\b(over|longer than|above)\s+2\s+hours\b", lowered):
            duration_min = 120

        semantic_query = query
        return StructuredMovieQuery(
            semantic_query=semantic_query,
            genres=genres,
            actors=actors,
            year_min=year_min,
            year_max=year_max,
            rating_min=rating_min,
            rating_max=rating_max,
            duration_min=duration_min,
            duration_max=duration_max,
            certificates=certificates,
        )

    @staticmethod
    def _heuristic_actors(query: str) -> list[str]:
        """Bat actor/cast names ro rang """
        name = r"[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){1,4}"
        stop = r"(?=\s+(?:or|and|from|in|with|rated|around|about)\b|[,.;?!]|$)"
        patterns = [
            rf"\b(?:featuring|starring)\s+({name}){stop}",
            rf"\bwith\s+(?:actor|actress|star|cast member)\s+({name}){stop}",
            rf"\babout\s+({name}){stop}",
        ]
        actors = []
        for pattern in patterns:
            for match in re.finditer(pattern, query):
                actor = " ".join(match.group(1).strip().split())
                if actor.casefold() not in {item.casefold() for item in actors}:
                    actors.append(actor)
        return actors

    def _merge(
        self,
        heuristic: StructuredMovieQuery,
        llm_structured: StructuredMovieQuery,
    ) -> StructuredMovieQuery:
        """Hop nhat ket qua LLM va heuristic, uu tien thong tin LLM khi co."""
        return StructuredMovieQuery(
            semantic_query=llm_structured.semantic_query or heuristic.semantic_query,
            genres=llm_structured.genres or heuristic.genres,
            actors=llm_structured.actors or heuristic.actors,
            year_min=llm_structured.year_min if llm_structured.year_min is not None else heuristic.year_min,
            year_max=llm_structured.year_max if llm_structured.year_max is not None else heuristic.year_max,
            rating_min=llm_structured.rating_min if llm_structured.rating_min is not None else heuristic.rating_min,
            rating_max=llm_structured.rating_max if llm_structured.rating_max is not None else heuristic.rating_max,
            duration_min=llm_structured.duration_min if llm_structured.duration_min is not None else heuristic.duration_min,
            duration_max=llm_structured.duration_max if llm_structured.duration_max is not None else heuristic.duration_max,
            certificates=llm_structured.certificates or heuristic.certificates,
        )
