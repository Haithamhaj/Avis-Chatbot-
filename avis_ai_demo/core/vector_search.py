from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from avis_ai_demo.core.data_lookup import load_kb
from avis_ai_demo.services.openai_service import OpenAIService


@dataclass(frozen=True)
class VectorSearchResult:
    source_kb: str
    record_id: str
    topic: str
    answer_ar: str
    answer_en: str
    score: float
    retrieval_method: str = "openai_embeddings"

    def as_general_faq_record(self) -> dict[str, Any]:
        return {
            "faq_id": self.record_id,
            "topic": self.topic,
            "risk_level": "low",
            "answer_ar": self.answer_ar,
            "answer_en": self.answer_en,
            "related_kbs": [self.source_kb],
            "semantic_score": self.score,
            "semantic_source": self.source_kb,
        }


@dataclass(frozen=True)
class KnowledgeChunk:
    source_kb: str
    record_id: str
    topic: str
    text: str
    answer_ar: str
    answer_en: str


def semantic_kb_search(
    query: str,
    service: OpenAIService | None,
    *,
    top_k: int = 1,
    threshold: float = 0.5,
    source_kbs: list[str] | None = None,
) -> list[VectorSearchResult]:
    if service is None:
        return []
    chunks = _knowledge_chunks()
    if source_kbs:
        allowed = set(source_kbs)
        chunks = tuple(chunk for chunk in chunks if chunk.source_kb in allowed)
    if not chunks:
        return []
    try:
        embeddings = service.embed_texts([query, *[chunk.text for chunk in chunks]])
    except Exception:
        return _lexical_kb_search(query, chunks, top_k=top_k)
    if len(embeddings) != len(chunks) + 1:
        return _lexical_kb_search(query, chunks, top_k=top_k)
    query_embedding = embeddings[0]
    scored = []
    for chunk, embedding in zip(chunks, embeddings[1:]):
        score = _cosine_similarity(query_embedding, embedding)
        if score >= threshold:
            scored.append(
                VectorSearchResult(
                    source_kb=chunk.source_kb,
                    record_id=chunk.record_id,
                    topic=chunk.topic,
                    answer_ar=chunk.answer_ar,
                    answer_en=chunk.answer_en,
                    score=round(score, 4),
                )
            )
    if scored:
        return sorted(scored, key=lambda item: (item.score, _source_priority(item.source_kb)), reverse=True)[:top_k]
    return _lexical_kb_search(query, chunks, top_k=top_k)


@lru_cache(maxsize=1)
def _knowledge_chunks() -> tuple[KnowledgeChunk, ...]:
    chunks: list[KnowledgeChunk] = []
    for record in load_kb("kb14-general-faq.json"):
        chunks.append(
            KnowledgeChunk(
                source_kb="KB14",
                record_id=record["faq_id"],
                topic=record["topic"],
                text=" ".join(
                    [
                        record.get("topic", ""),
                        " ".join(record.get("aliases") or []),
                        record.get("answer_ar", ""),
                        record.get("answer_en", ""),
                    ]
                ),
                answer_ar=record["answer_ar"],
                answer_en=record["answer_en"],
            )
        )

    for record in load_kb("kb06-booking-channels.json"):
        chunks.append(
            KnowledgeChunk(
                source_kb="KB06",
                record_id=record["channel_id"],
                topic="booking_channels",
                text=" ".join(
                    [
                        record.get("platform", ""),
                        record.get("availability", ""),
                        " ".join(record.get("supported_services") or []),
                        record.get("details_ar", ""),
                        record.get("details_en", ""),
                    ]
                ),
                answer_ar=record["details_ar"],
                answer_en=record["details_en"],
            )
        )

    chauffeur = load_kb("kb11-chauffeur.json")
    chunks.append(
        KnowledgeChunk(
            source_kb="KB11",
            record_id="chauffeur_service",
            topic="chauffeur_service",
            text=" ".join(
                [
                    chauffeur.get("service_name", ""),
                    "سواق سائق سيارة مع سواق سيارة مع سائق chauffeur driver car with driver",
                    " ".join(chauffeur.get("available_cities") or []),
                    " ".join(chauffeur.get("not_available_cities") or []),
                    chauffeur.get("answer_ar", ""),
                    chauffeur.get("answer_en", ""),
                    chauffeur.get("not_available_response_ar", ""),
                    chauffeur.get("not_available_response_en", ""),
                ]
            ),
            answer_ar=chauffeur["answer_ar"],
            answer_en=chauffeur["answer_en"],
        )
    )

    for record in load_kb("kb12-corporate.json"):
        chunks.append(
            KnowledgeChunk(
                source_kb="KB12",
                record_id=record["service_id"],
                topic="corporate_services",
                text=" ".join(
                    [
                        record.get("name_ar", ""),
                        record.get("name_en", ""),
                        record.get("description_ar", ""),
                        record.get("description_en", ""),
                        record.get("target", ""),
                    ]
                ),
                answer_ar=record["description_ar"],
                answer_en=record["description_en"],
            )
        )
    return tuple(chunks)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _lexical_kb_search(
    query: str,
    chunks: tuple[KnowledgeChunk, ...],
    *,
    top_k: int,
) -> list[VectorSearchResult]:
    query_terms = _terms(query)
    if not query_terms:
        return []
    scored = []
    for chunk in chunks:
        chunk_terms = _terms(chunk.text)
        overlap = query_terms & chunk_terms
        if overlap:
            score = len(overlap) / max(len(query_terms), 1)
            scored.append(
                VectorSearchResult(
                    source_kb=chunk.source_kb,
                    record_id=chunk.record_id,
                    topic=chunk.topic,
                    answer_ar=chunk.answer_ar,
                    answer_en=chunk.answer_en,
                    score=round(score, 4),
                    retrieval_method="lexical_fallback",
                )
            )
    return sorted(scored, key=lambda item: (item.score, _source_priority(item.source_kb)), reverse=True)[:top_k]


def _terms(text: str) -> set[str]:
    normalized = text.lower()
    for char in "؟?.,،:;/()[]{}":
        normalized = normalized.replace(char, " ")
    return {term for term in normalized.split() if len(term) > 2}


def _source_priority(source_kb: str) -> int:
    priorities = {
        "KB11": 4,
        "KB12": 4,
        "KB06": 3,
        "KB14": 1,
    }
    return priorities.get(source_kb, 2)
