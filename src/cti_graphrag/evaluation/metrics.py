"""Metrics for answer quality, retrieval quality, and RAG-specific quality."""

from __future__ import annotations

import math
import re

from cti_graphrag.embeddings import Embedder, cosine_similarity, tokenize
from cti_graphrag.verification.claim_extraction import extract_claims

# ---------------------------------------------------------------------------
# Answer quality
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if _normalize_text(prediction) == _normalize_text(gold) else 0.0


def token_f1(prediction: str, gold: str) -> dict[str, float]:
    pred_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        equal = pred_tokens == gold_tokens
        return {"precision": float(equal), "recall": float(equal), "f1": float(equal)}

    common: dict[str, int] = {}
    for tok in pred_tokens:
        if tok in gold_tokens:
            common[tok] = common.get(tok, 0) + 1
    num_same = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in set(common))

    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def semantic_similarity(prediction: str, gold: str, embedder: Embedder) -> float:
    vecs = embedder.embed([prediction, gold])
    return float(cosine_similarity(vecs[0], vecs[1:2])[0])


def entity_coverage(prediction: str, relevant_entities: list[str]) -> float:
    """Fraction of ground-truth relevant entities actually mentioned in the answer."""
    if not relevant_entities:
        return 1.0
    pred_lower = prediction.lower()
    hits = sum(1 for e in relevant_entities if e.lower() in pred_lower)
    return hits / len(relevant_entities)


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """Multiple chunks can come from the same source document; retrieval metrics
    should count that document once, not once per chunk."""
    seen: set[str] = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = _dedupe_preserve_order(retrieved)[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in relevant) / len(top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    top = _dedupe_preserve_order(retrieved)[:k]
    return sum(1 for r in top if r in relevant) / len(relevant)


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(_dedupe_preserve_order(retrieved)):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = _dedupe_preserve_order(retrieved)[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(top) if r in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# RAG-specific quality
# ---------------------------------------------------------------------------


def citation_correctness(cited_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    """Fraction of cited documents that are actually among the expected supporting documents."""
    if not cited_doc_ids:
        return 0.0
    expected = set(expected_doc_ids)
    correct = sum(1 for d in cited_doc_ids if d in expected)
    return correct / len(cited_doc_ids)


def answer_relevance(prediction: str, question: str, embedder: Embedder) -> float:
    return semantic_similarity(prediction, question, embedder)


def context_relevance(retrieved_texts: list[str], question: str, embedder: Embedder) -> float:
    if not retrieved_texts:
        return 0.0
    question_tokens = set(tokenize(question))
    scores = []
    for text in retrieved_texts:
        doc_tokens = set(tokenize(text))
        scores.append(len(question_tokens & doc_tokens) / max(len(question_tokens), 1))
    return sum(scores) / len(scores)


def faithfulness(answer_text: str, evidence_texts: list[str], threshold: float = 0.25) -> float:
    """Fraction of answer claims lexically grounded in the supplied evidence texts.

    Works directly off plain evidence strings (graph path text + cited chunk
    text) so it can score any system's output uniformly, independent of the
    internal verification pass ``AgenticGraphRAG`` runs on itself.
    """
    claims = extract_claims(answer_text)
    if not claims:
        return 1.0

    evidence_tokens: set[str] = set()
    for text in evidence_texts:
        evidence_tokens |= set(tokenize(text))

    supported = 0
    for claim in claims:
        claim_tokens = set(tokenize(claim))
        if len(claim_tokens) < 3:
            supported += 1
            continue
        overlap = len(claim_tokens & evidence_tokens) / max(len(claim_tokens), 1)
        if overlap >= threshold:
            supported += 1
    return supported / len(claims)
