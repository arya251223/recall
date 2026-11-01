"""RAG pipeline: guard -> embed -> (retrieve) -> semantic cache -> LLM -> safety -> cache."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .cache import SemanticCache
from .embeddings import Embedder
from .guardrails import check_input, grounding_score, sanitize_output
from .knowledge_base import KnowledgeBase
from .metrics import Metrics

NO_CONTEXT_MSG = "I couldn't find anything about that in the knowledge base."
_DONT_KNOW = re.compile(r"don'?t know|do not know|not (mentioned|specified|provided|available)|cannot find|can't find", re.I)


@dataclass
class PipelineConfig:
    top_k: int = 3
    min_relevance: float = 0.12
    consistency_check: bool = True
    min_grounding: float = 0.30
    invalidate_similarity: float = 0.30


class Pipeline:
    def __init__(self, embedder: Embedder, kb: KnowledgeBase, cache: SemanticCache, llm,
                 config: PipelineConfig | None = None, metrics: Metrics | None = None):
        self.embedder, self.kb, self.cache, self.llm = embedder, kb, cache, llm
        self.cfg = config or PipelineConfig(min_relevance=embedder.default_min_relevance,
                                            invalidate_similarity=embedder.default_invalidate_similarity)
        self.metrics = metrics or Metrics()

    # -- knowledge-base mutations keep the cache honest ----------------------------
    def add_document(self, title: str, text: str, doc_id: str | None = None) -> dict:
        doc, vecs = self.kb.add_document(title, text, doc_id)
        dropped = self.cache.invalidate_docs({doc.id}) + self.cache.invalidate_similar(vecs, self.cfg.invalidate_similarity)
        return {"id": doc.id, "title": doc.title, "chunks": doc.n_chunks, "cache_entries_invalidated": dropped}

    def remove_document(self, doc_id: str) -> dict:
        ok = self.kb.remove_document(doc_id)
        return {"removed": ok, "cache_entries_invalidated": self.cache.invalidate_docs({doc_id}) if ok else 0}

    # -- query ---------------------------------------------------------------------
    def query(self, question: str, use_cache: bool = True) -> dict:
        t0 = time.perf_counter()
        trace: list[dict] = []
        out = {"question": question, "answer": "", "source": "llm", "similarity": None, "matched_query": None,
               "sources": [], "rejected": [], "flags": {}, "tokens": {"prompt": 0, "completion": 0, "saved": 0},
               "model": getattr(self.llm, "model", "?"), "kb_version": self.kb.version}

        def stage(name: str, status: str, t_start: float | None = None):
            trace.append({"stage": name, "status": status,
                          "ms": round((time.perf_counter() - t_start) * 1000, 3) if t_start else 0.0})

        def finish(source: str, tokens_spent: int = 0) -> dict:
            total = (time.perf_counter() - t0) * 1000
            out["source"], out["trace"], out["total_ms"] = source, trace, round(total, 3)
            self.metrics.record(source, total, out["tokens"]["saved"], tokens_spent, question)
            return out

        # 1. input guard
        t = time.perf_counter()
        g = check_input(question)
        stage("guard", "done" if g.ok else "blocked", t)
        if not g.ok:
            out["answer"] = g.reason
            for s in ("embed", "cache", "retrieve", "generate", "safety"):
                stage(s, "skipped")
            return finish("blocked")
        question = question.strip()

        # 2. embed once, reuse for cache + retrieval
        t = time.perf_counter()
        qvec = self.embedder.encode([question])[0]
        stage("embed", "done", t)

        hits = None
        doc_ids = None
        if use_cache and self.cfg.consistency_check:
            t = time.perf_counter()
            hits = self.kb.search(qvec, self.cfg.top_k)
            doc_ids = list(dict.fromkeys(h.chunk.doc_id for h in hits))  # best first, unique
            retrieve_ms = (time.perf_counter() - t) * 1000

        # 3. semantic cache
        if use_cache:
            t = time.perf_counter()
            res = self.cache.lookup(qvec, question, retrieved_doc_ids=doc_ids)
            stage("cache", "hit" if res.hit else "miss", t)
            out["rejected"] = res.rejected
            out["cache_reason"] = res.reason
            if res.hit:
                e = res.entry
                out.update(answer=e.answer, similarity=round(res.score, 4), matched_query=e.query, sources=e.sources)
                out["tokens"]["saved"] = e.tokens
                if hits is not None:
                    trace.append({"stage": "retrieve", "status": "done", "ms": round(retrieve_ms, 3)})
                else:
                    stage("retrieve", "skipped")
                stage("generate", "skipped")
                stage("safety", "skipped")
                return finish("cache")
        else:
            stage("cache", "skipped")

        # 4. retrieve
        if hits is None:
            t = time.perf_counter()
            hits = self.kb.search(qvec, self.cfg.top_k)
            stage("retrieve", "done", t)
        else:
            trace.append({"stage": "retrieve", "status": "done", "ms": round(retrieve_ms, 3)})
        out["sources"] = [{"doc_id": h.chunk.doc_id, "title": h.chunk.doc_title, "text": h.chunk.text,
                           "score": round(h.score, 4)} for h in hits]

        if not hits or hits[0].score < self.cfg.min_relevance:
            out["answer"], out["sources"] = NO_CONTEXT_MSG, []
            stage("generate", "skipped")
            stage("safety", "skipped")
            return finish("no_context")

        # 5. generate
        t = time.perf_counter()
        try:
            res_llm = self.llm.generate(question, hits)
        except Exception as exc:  # noqa: BLE001
            stage("generate", "error", t)
            stage("safety", "skipped")
            out["answer"] = f"The language model call failed: {exc}"
            return finish("error")
        stage("generate", "done", t)
        out["model"] = res_llm.model
        out["tokens"].update(prompt=res_llm.prompt_tokens, completion=res_llm.completion_tokens)

        # 6. output safety
        t = time.perf_counter()
        answer, redactions = sanitize_output(res_llm.text)
        ground = grounding_score(answer, " ".join(h.chunk.text for h in hits))
        stage("safety", "done", t)
        out["answer"] = answer
        out["flags"] = {"redactions": redactions, "grounding": round(ground, 3)}

        # 7. cache only good answers
        cacheable = ground >= self.cfg.min_grounding and not _DONT_KNOW.search(answer)
        out["flags"]["cached"] = cacheable and use_cache
        if cacheable and use_cache:
            self.cache.add(qvec, question, answer, out["sources"], res_llm.prompt_tokens + res_llm.completion_tokens)
        return finish("llm", res_llm.prompt_tokens + res_llm.completion_tokens)

    def config_view(self) -> dict:
        return {"cache_threshold": self.cache.threshold, "entity_guard": self.cache.entity_guard,
                "consistency_check": self.cfg.consistency_check, "top_k": self.cfg.top_k,
                "min_relevance": self.cfg.min_relevance, "embedder": self.embedder.name,
                "llm": getattr(self.llm, "model", "?"), "kb_docs": len(self.kb.docs), "kb_version": self.kb.version,
                "cache_entries": len(self.cache), "cache_ttl_s": self.cache.ttl, "cache_max": self.cache.max_entries}


def build_pipeline(settings, llm=None) -> Pipeline:
    from .embeddings import build_embedder
    from .llm import build_llm

    emb = build_embedder(settings.embedder)
    kb = KnowledgeBase(emb)
    if settings.kb_dir and settings.kb_dir.exists():
        kb.load_dir(settings.kb_dir)
    thr = settings.cache_threshold if settings.cache_threshold is not None else emb.default_cache_threshold
    cache = SemanticCache(emb.dim, thr, settings.cache_ttl_seconds, settings.cache_max_entries)
    cfg = PipelineConfig(top_k=settings.top_k, min_relevance=emb.default_min_relevance,
                         invalidate_similarity=emb.default_invalidate_similarity)
    return Pipeline(emb, kb, cache, llm or build_llm(settings), cfg)
