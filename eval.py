import asyncio
import httpx
import json
from dataclasses import dataclass, asdict

BASE_URL = "http://localhost:8000"

# ── golden test set ────────────────────────────────────────────────────────────
PYTHON_QUERIES = [
    "How do I read a CSV file with pandas?",
    "How to reverse a list in Python?",
    "How do I handle exceptions in Python?",
    "What is the difference between @staticmethod and @classmethod?",
    "How do I use async and await in Python?",
    "How do Python decorators work?",
    "What is the fastest way to check if a key exists in a dictionary?",
    "How do I use list comprehension with conditions?",
]

OFF_TOPIC_QUERIES = [
    "What is the capital of France?",
    "Who won the FIFA World Cup in 2022?",
    "How do I make pasta carbonara?",
]


@dataclass
class EvalResult:
    query: str
    is_off_topic_test: bool
    retrieval_relevance: float   # mean similarity of retrieved chunks
    answer_length: int           # proxy for completeness
    faithfulness_score: float    # lexical overlap proxy
    correctly_rejected: bool     # only meaningful for off-topic queries
    latency_ms: int
    confidence: float


def lexical_faithfulness(answer: str, sources: list[dict]) -> float:
    """
    Proxy faithfulness: what fraction of source titles' keywords appear in the answer?
    Real faithfulness would use an LLM judge (e.g. Ragas), but this runs offline.
    """
    if not sources:
        return 0.0
    answer_lower = answer.lower()
    hits = 0
    total = 0
    for src in sources:
        keywords = [w for w in src["title"].lower().split() if len(w) > 3]
        total += len(keywords)
        hits += sum(1 for kw in keywords if kw in answer_lower)
    return round(hits / total, 3) if total else 0.0


async def evaluate_query(client: httpx.AsyncClient, query: str, off_topic: bool) -> EvalResult:
    resp = await client.post(
        f"{BASE_URL}/ask",
        json={"question": query},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    sources = data.get("sources", [])
    retrieval_relevance = (
        sum(s["relevance"] for s in sources) / len(sources) if sources else 0.0
    )
    faithfulness = lexical_faithfulness(data.get("answer", ""), sources)
    correctly_rejected = off_topic and data.get("off_topic", False)

    return EvalResult(
        query=query,
        is_off_topic_test=off_topic,
        retrieval_relevance=round(retrieval_relevance, 3),
        answer_length=len(data.get("answer", "")),
        faithfulness_score=faithfulness,
        correctly_rejected=correctly_rejected,
        latency_ms=data.get("latency_ms", 0),
        confidence=data.get("confidence", 0.0),
    )


async def run_eval():
    results: list[EvalResult] = []

    async with httpx.AsyncClient() as client:
        # Check health first
        health = await client.get(f"{BASE_URL}/health")
        print(f"✅ Service status: {health.json()['status']}\n")

        tasks = (
            [evaluate_query(client, q, False) for q in PYTHON_QUERIES]
            + [evaluate_query(client, q, True) for q in OFF_TOPIC_QUERIES]
        )
        results = await asyncio.gather(*tasks)

    # ── summary ────────────────────────────────────────────────────────────────
    python_results = [r for r in results if not r.is_off_topic_test]
    off_topic_results = [r for r in results if r.is_off_topic_test]

    avg_relevance = sum(r.retrieval_relevance for r in python_results) / len(python_results)
    avg_faithfulness = sum(r.faithfulness_score for r in python_results) / len(python_results)
    avg_latency = sum(r.latency_ms for r in python_results) / len(python_results)
    rejection_rate = sum(1 for r in off_topic_results if r.correctly_rejected) / len(off_topic_results)

    print("=" * 60)
    print("EVAL RESULTS — Python Q&A RAG Pipeline")
    print("=" * 60)
    print(f"{'Metric':<35} {'Score':>10}")
    print("-" * 60)
    print(f"{'Avg retrieval relevance (0–1)':<35} {avg_relevance:>10.3f}")
    print(f"{'Avg answer faithfulness (0–1)':<35} {avg_faithfulness:>10.3f}")
    print(f"{'Avg latency (ms)':<35} {avg_latency:>10.0f}")
    print(f"{'Off-topic rejection rate':<35} {rejection_rate:>10.0%}")
    print("=" * 60)

    print("\nPer-query breakdown (Python questions):")
    for r in python_results:
        print(
            f"  [{r.latency_ms:>4}ms | rel={r.retrieval_relevance:.2f} | "
            f"faith={r.faithfulness_score:.2f}] {r.query[:60]}"
        )

    print("\nOff-topic queries:")
    for r in off_topic_results:
        status = "✅ REJECTED" if r.correctly_rejected else "❌ NOT REJECTED"
        print(f"  {status} — {r.query}")

    # Save JSON for notebook/README
    with open("eval_results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print("\n📄 Full results saved to eval_results.json")


if __name__ == "__main__":
    asyncio.run(run_eval())
