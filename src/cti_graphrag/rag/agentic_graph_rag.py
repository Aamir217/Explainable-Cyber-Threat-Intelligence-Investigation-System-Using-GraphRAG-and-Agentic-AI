"""System 4 -- Agentic GraphRAG: Planner -> Graph/Vector/CVE tools -> Verification -> LLM.

Unlike ``GraphRAGSystem`` (a fixed pipeline that always runs both
retrievers), this system:

1. Asks the planner which tools the question actually needs and how deep to
   traverse the graph.
2. Runs a second, deeper retrieval iteration if the first pass returns no
   evidence at all (adaptive retrieval).
3. Generates a draft answer, then runs it through the verification agent,
   producing a final answer restricted to claims grounded in the retrieved
   evidence.
"""

from __future__ import annotations

from cti_graphrag.agents.planner import plan
from cti_graphrag.agents.tools import DEFAULT_TOOLS, fuse_chunks
from cti_graphrag.corpus import Corpus
from cti_graphrag.rag.base import AnswerResult, BaseRAGSystem
from cti_graphrag.verification.verifier import grounded_answer, verify_answer


class AgenticGraphRAG(BaseRAGSystem):
    system_name = "agentic_graph_rag"

    def __init__(self, corpus: Corpus, llm, top_k: int = 5, verify: bool = True, name: str | None = None):
        super().__init__(llm)
        self._corpus = corpus
        self._top_k = top_k
        self._verify = verify
        if name:
            self.system_name = name

    def _run_tools(self, question: str, tool_plan, graph_hops: int):
        tool_calls: list[str] = []
        graph_paths = []
        tool_results = []

        for tool_name in tool_plan.tools:
            tool = DEFAULT_TOOLS[tool_name]
            kwargs = {"max_hops": graph_hops} if tool_name == "graph_search" else {}
            if tool_name == "cve_lookup":
                kwargs["cve_ids"] = tool_plan.cve_ids
            result = tool.run(question, self._corpus, **kwargs)
            tool_calls.append(tool_name)
            tool_results.append(result)
            graph_paths.extend(result.graph_paths)

        fused_chunks = fuse_chunks(*tool_results)
        evidence = self._corpus.reranker.rerank(question, fused_chunks, top_k=self._top_k) if fused_chunks else []
        return tool_calls, graph_paths, evidence

    def _answer_impl(self, question: str) -> AnswerResult:
        tool_plan = plan(question, self._corpus)
        tool_calls, graph_paths, evidence = self._run_tools(question, tool_plan, tool_plan.graph_hops)

        iterations = 1
        if not graph_paths and not evidence:
            # Adaptive retrieval: nothing came back, so widen the search
            # (deeper graph hops, larger candidate pool) instead of giving up.
            deeper_hops = min(4, tool_plan.graph_hops + 2)
            tool_calls_2, graph_paths, evidence = self._run_tools(question, tool_plan, deeper_hops)
            tool_calls.extend(f"retry:{t}" for t in tool_calls_2)
            iterations = 2

        draft_answer = self._generate(question, graph_paths=graph_paths, evidence=evidence)

        metadata = {"planner_rationale": tool_plan.rationale, "retrieval_iterations": iterations}

        if self._verify:
            report = verify_answer(draft_answer, graph_paths, evidence)
            final_answer = grounded_answer(draft_answer, report)
            metadata["faithfulness"] = report.faithfulness
            metadata["unsupported_claims"] = report.unsupported_claims
            tool_calls.append("verification")
        else:
            final_answer = draft_answer

        return AnswerResult(
            system_name=self.system_name,
            question=question,
            answer=final_answer,
            citations=self._to_citations(evidence),
            graph_paths=graph_paths,
            retrieved_chunk_ids=[sc.chunk.chunk_id for sc in evidence],
            tool_calls=tool_calls,
            metadata=metadata,
        )
