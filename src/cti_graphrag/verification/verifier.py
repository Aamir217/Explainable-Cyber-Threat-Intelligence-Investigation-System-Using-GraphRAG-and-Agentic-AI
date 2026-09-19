"""Evidence verification: check whether each claim in a generated answer is grounded.

This implements the "Verification Agent" from the architecture: before a
response is returned, every sentence of the draft answer is checked against
the retrieved graph paths and document chunks. Claims with no lexical
grounding in the evidence are flagged (and, in ``AgenticGraphRAG``, stripped
from the final answer) rather than passed through -- directly addressing the
hallucination and unsupported-claim failure modes called out in the
project's error-analysis taxonomy.

The check is intentionally a transparent lexical-overlap heuristic (no
second LLM call, no external NLI model) so it stays fast, free, and fully
deterministic for evaluation; a production deployment could swap this for an
LLM-as-judge or NLI entailment model behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cti_graphrag.embeddings import tokenize
from cti_graphrag.graph.store import GraphPath
from cti_graphrag.retrieval.vector_store import ScoredChunk
from cti_graphrag.verification.claim_extraction import extract_claims

SUPPORT_THRESHOLD = 0.25


@dataclass
class ClaimVerification:
    claim: str
    supported: bool
    best_overlap: float
    evidence_ref: str | None


@dataclass
class VerificationReport:
    claims: list[ClaimVerification] = field(default_factory=list)

    @property
    def faithfulness(self) -> float:
        if not self.claims:
            return 1.0
        return sum(1 for c in self.claims if c.supported) / len(self.claims)

    @property
    def unsupported_claims(self) -> list[str]:
        return [c.claim for c in self.claims if not c.supported]


def _evidence_texts(graph_paths: list[GraphPath], chunks: list[ScoredChunk]) -> list[tuple[str, str]]:
    texts = [(f"graph:{p.to_text()}", p.to_text()) for p in graph_paths]
    texts += [(f"chunk:{sc.chunk.chunk_id}", sc.chunk.text) for sc in chunks]
    return texts


def _overlap(claim_tokens: set[str], evidence_tokens: set[str]) -> float:
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def verify_answer(
    answer_text: str,
    graph_paths: list[GraphPath],
    chunks: list[ScoredChunk],
    threshold: float = SUPPORT_THRESHOLD,
) -> VerificationReport:
    evidence_texts = _evidence_texts(graph_paths, chunks)
    evidence_tokens = [(ref, set(tokenize(text))) for ref, text in evidence_texts]

    report = VerificationReport()
    for claim in extract_claims(answer_text):
        claim_tokens = set(tokenize(claim))
        # Ignore boilerplate/meta sentences (e.g. "No evidence was found...")
        # which are true statements about the system, not factual claims.
        if len(claim_tokens) < 3:
            report.claims.append(ClaimVerification(claim, supported=True, best_overlap=1.0, evidence_ref=None))
            continue

        best_ref, best_score = None, 0.0
        for ref, tokens in evidence_tokens:
            score = _overlap(claim_tokens, tokens)
            if score > best_score:
                best_ref, best_score = ref, score

        report.claims.append(
            ClaimVerification(
                claim=claim,
                supported=best_score >= threshold,
                best_overlap=best_score,
                evidence_ref=best_ref if best_score >= threshold else None,
            )
        )
    return report


def grounded_answer(answer_text: str, report: VerificationReport) -> str:
    """Rebuild the answer keeping only verified claims, noting anything dropped."""
    supported = [c.claim for c in report.claims if c.supported]
    if not supported:
        return "Insufficient verified evidence to answer this question confidently."

    text = " ".join(supported)
    if report.unsupported_claims:
        text += f" (Note: {len(report.unsupported_claims)} additional unverified statement(s) were withheld.)"
    return text
