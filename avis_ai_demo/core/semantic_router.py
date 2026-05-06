from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from avis_ai_demo.core.data_lookup import load_kb
from avis_ai_demo.core.types import Intent
from avis_ai_demo.services.openai_service import OpenAIService


@dataclass(frozen=True)
class SemanticRouteCandidate:
    route_id: str
    route_type: str
    target_intent: Intent
    target_modules: list[str]
    score: float
    retrieval_method: str
    allowed_next_step: str

    def to_trace(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "type": self.route_type,
            "target_intent": self.target_intent.value,
            "target_modules": list(self.target_modules),
            "score": self.score,
            "provider": self.retrieval_method,
            "allowed_next_step": self.allowed_next_step,
        }


@dataclass(frozen=True)
class RouteMetadata:
    route_id: str
    route_type: str
    target_intent: Intent
    target_modules: list[str]
    text: str
    allowed_next_step: str


def semantic_route_candidates(
    query: str,
    service: OpenAIService | None,
    *,
    top_k: int = 3,
    threshold: float = 0.48,
) -> list[SemanticRouteCandidate]:
    routes = _route_metadata()
    if service is None or not routes:
        return []
    try:
        embeddings = service.embed_texts([query, *[route.text for route in routes]])
    except Exception:
        return _lexical_route_candidates(query, routes, top_k=top_k)
    if len(embeddings) != len(routes) + 1:
        return _lexical_route_candidates(query, routes, top_k=top_k)

    query_embedding = embeddings[0]
    candidates = []
    for route, embedding in zip(routes, embeddings[1:]):
        score = _cosine_similarity(query_embedding, embedding)
        if score >= threshold:
            candidates.append(_candidate(route, score, "openai_embeddings"))
    if candidates:
        return sorted(candidates, key=lambda item: (item.score, _route_priority(item)), reverse=True)[:top_k]
    return _lexical_route_candidates(query, routes, top_k=top_k)


def should_use_semantic_candidate(current_intent: Intent, candidate: SemanticRouteCandidate | None) -> bool:
    if candidate is None:
        return False
    if current_intent == Intent.FALLBACK_UNKNOWN:
        return True
    if current_intent == Intent.GENERAL_FAQ and candidate.target_intent == Intent.GENERAL_FAQ:
        return True
    return False


@lru_cache(maxsize=1)
def _route_metadata() -> tuple[RouteMetadata, ...]:
    routes = []
    for record in load_kb("kb16-semantic-route-metadata.json"):
        routes.append(
            RouteMetadata(
                route_id=record["route_id"],
                route_type=record["type"],
                target_intent=Intent(record["target_intent"]),
                target_modules=list(record.get("target_modules") or []),
                text=" ".join(
                    [
                        record["route_id"],
                        record["type"],
                        record["target_intent"],
                        " ".join(record.get("target_modules") or []),
                        " ".join(record.get("examples_ar") or []),
                        " ".join(record.get("examples_en") or []),
                        " ".join(record.get("required_validation") or []),
                        record.get("allowed_next_step", ""),
                        record.get("notes", ""),
                    ]
                ),
                allowed_next_step=record.get("allowed_next_step", "deterministic_router"),
            )
        )
    return tuple(routes)


def _lexical_route_candidates(
    query: str,
    routes: tuple[RouteMetadata, ...],
    *,
    top_k: int,
) -> list[SemanticRouteCandidate]:
    query_terms = _terms(query)
    if not query_terms:
        return []
    candidates = []
    for route in routes:
        route_terms = _terms(route.text)
        overlap = query_terms & route_terms
        if overlap:
            score = len(overlap) / max(len(query_terms), 1)
            candidates.append(_candidate(route, score, "lexical_fallback"))
    return sorted(candidates, key=lambda item: (item.score, _route_priority(item)), reverse=True)[:top_k]


def _candidate(route: RouteMetadata, score: float, retrieval_method: str) -> SemanticRouteCandidate:
    return SemanticRouteCandidate(
        route_id=route.route_id,
        route_type=route.route_type,
        target_intent=route.target_intent,
        target_modules=route.target_modules,
        score=round(score, 4),
        retrieval_method=retrieval_method,
        allowed_next_step=route.allowed_next_step,
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _terms(text: str) -> set[str]:
    normalized = text.lower()
    for char in "؟?.,،:;/()[]{}":
        normalized = normalized.replace(char, " ")
    return {term for term in normalized.split() if len(term) > 2}


def _route_priority(candidate: SemanticRouteCandidate) -> int:
    route_priorities = {
        "roadside_assistance_workflow": 5,
        "deposit_policy_or_dispute": 5,
        "daily_rental_workflow": 4,
        "monthly_rental_workflow": 4,
        "chauffeur_service_info": 4,
        "corporate_services_info": 4,
        "branch_lookup": 4,
        "fleet_price_lookup": 4,
        "booking_channels_info": 3,
        "brand_company_info": 1,
    }
    return route_priorities.get(candidate.route_id, 2)
