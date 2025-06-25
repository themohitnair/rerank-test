import asyncio
from qdrant_client import AsyncQdrantClient
import pandas as pd
import numpy as np
import requests
import os
import time
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagReranker

load_dotenv()

client = AsyncQdrantClient(host="localhost", port=6333)
JINA_API_KEY = os.getenv("JINA_API_KEY")

MODELS = {
    "jina_v3": {
        "name": "jina-embeddings-v3",
        "dim": 1024,
        "color": "mediumpurple",
        "type": "jina_api",
    },
}

RERANKERS = {
    "mxbai_large": {
        "name": "mixedbread-ai/mxbai-rerank-large-v1",
        "type": "sentence_transformer",
        "color": "orange",
    },
    "mxbai_base": {
        "name": "mixedbread-ai/mxbai-rerank-base-v1",
        "type": "sentence_transformer",
        "color": "darkorange",
    },
    "jina_v2_multilingual": {
        "name": "jina-reranker-v2-base-multilingual",
        "type": "jina_api",
        "color": "purple",
    },
    "bge_base": {
        "name": "BAAI/bge-reranker-base",
        "type": "flag_embedding",
        "color": "lightgreen",
    },
    "bge_large": {
        "name": "BAAI/bge-reranker-large",
        "type": "flag_embedding",
        "color": "darkgreen",
    },
    "msmarco_minilm": {
        "name": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "type": "sentence_transformer",
        "color": "lightblue",
    },
}

SPECIFIC_TOPICS = [
    "England Cricket Team",
    "England Football Team",
    "Black Coffee",
    "Indian Cricket Team",
    "Black History Month",
]


async def get_topic_counts(collection_name):
    topic_counts = {}
    scroll_result = await client.scroll(
        collection_name=collection_name, limit=10000, with_payload=True
    )
    all_points = scroll_result[0]
    for point in all_points:
        topic = point.payload.get("topic", "Unknown")
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    print(f"Actual topic distribution in {collection_name}:")
    for topic, count in topic_counts.items():
        print(f"  {topic}: {count} documents")
    return topic_counts


def get_jina_embedding(text, model_name):
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
    }
    data = {
        "model": model_name,
        "normalized": True,
        "embedding_type": "float",
        "input": [text],
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def jina_rerank(
    query, documents, model="jina-reranker-v2-base-multilingual", top_n=None
):
    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
    }
    data = {"model": model, "query": query, "documents": documents, "top_n": top_n}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()


def flag_rerank(query, documents, model_name, top_n=None):
    try:
        reranker = FlagReranker(model_name, use_fp16=True)
        pairs = [[query, doc] for doc in documents]
        scores = reranker.compute_score(pairs)
        results = []
        for i, score in enumerate(scores):
            results.append({"index": i, "relevance_score": float(score)})
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        if top_n:
            results = results[:top_n]
        return {"results": results}
    except Exception as e:
        print(f"FlagEmbedding reranking failed: {e}")
        results = [{"index": i, "relevance_score": 0.0} for i in range(len(documents))]
        if top_n:
            results = results[:top_n]
        return {"results": results}


def sentence_transformer_rerank(query, documents, model_name, top_n=None):
    try:
        cross_encoder = CrossEncoder(model_name)
        pairs = [[query, doc] for doc in documents]
        scores = cross_encoder.predict(pairs)
        results = []
        for i, score in enumerate(scores):
            results.append({"index": i, "relevance_score": float(score)})
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        if top_n:
            results = results[:top_n]
        return {"results": results}
    except Exception as e:
        print(f"SentenceTransformer reranking failed: {e}")
        results = [{"index": i, "relevance_score": 0.0} for i in range(len(documents))]
        if top_n:
            results = results[:top_n]
        return {"results": results}


def rerank_documents(query, documents, reranker_info, top_n=None):
    reranker_type = reranker_info["type"]
    model_name = reranker_info["name"]
    start_time = time.time()
    try:
        if reranker_type == "jina_api":
            result = jina_rerank(query, documents, model_name, top_n)
        elif reranker_type == "flag_embedding":
            result = flag_rerank(query, documents, model_name, top_n)
        elif reranker_type == "sentence_transformer":
            result = sentence_transformer_rerank(query, documents, model_name, top_n)
        else:
            raise ValueError(f"Unknown reranker type: {reranker_type}")
    except Exception as e:
        print(f"Reranking failed for {model_name}: {e}")
        result = {
            "results": [
                {"index": i, "relevance_score": 0.0}
                for i in range(min(top_n or len(documents), len(documents)))
            ]
        }
    end_time = time.time()
    rerank_time = end_time - start_time
    return result, rerank_time


async def evaluate_rerankers(
    model_key, model_info, k=100, score_threshold=0.3, rerank_top_percent=0.8
):
    results_by_reranker = {}
    original_results = {}
    timing_data = {}
    collection_name = f"{model_key}_posts"
    topic_counts = await get_topic_counts(collection_name)
    for topic in SPECIFIC_TOPICS:
        print(f"Querying '{topic}' in {collection_name}...")
        total_relevant_docs = topic_counts.get(topic, 0)
        print(f"  Total relevant documents in collection: {total_relevant_docs}")
        query_vector = get_jina_embedding(topic, model_info["name"])
        query_results = await client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
            score_threshold=score_threshold,
        )
        if not query_results.points:
            print(f"  No results found for '{topic}'")
            for reranker_key in RERANKERS.keys():
                if reranker_key not in results_by_reranker:
                    results_by_reranker[reranker_key] = {}
                    timing_data[reranker_key] = {}
                results_by_reranker[reranker_key][topic] = {
                    "true_positives": 0,
                    "false_positives": 0,
                    "precision": 0,
                    "recall": 0,
                    "retrieved_topics": [],
                    "total_retrieved": 0,
                    "reranked_total": 0,
                    "original_total": 0,
                    "total_relevant_in_collection": total_relevant_docs,
                }
                timing_data[reranker_key][topic] = 0.0
            continue
        original_total = len(query_results.points)
        target_rerank_count = int(len(query_results.points) * rerank_top_percent)
        original_topics = [
            point.payload["topic"]
            for point in query_results.points[:target_rerank_count]
        ]
        original_tp = sum(1 for t in original_topics if t == topic)
        original_precision = (
            original_tp / len(original_topics) if len(original_topics) > 0 else 0
        )
        original_recall = (
            original_tp / total_relevant_docs if total_relevant_docs > 0 else 0
        )
        original_results[topic] = {
            "true_positives": original_tp,
            "false_positives": len(original_topics) - original_tp,
            "precision": original_precision,
            "recall": original_recall,
            "retrieved_topics": original_topics,
            "total_retrieved": len(original_topics),
            "total_relevant_in_collection": total_relevant_docs,
        }
        documents = [point.payload["post"] for point in query_results.points]
        print(f"  Initial retrieval: {original_total} documents")
        print(f"  Target rerank count: {target_rerank_count}")
        print(f"  Original precision@{target_rerank_count}: {original_precision:.3f}")
        print(f"  Original recall@{target_rerank_count}: {original_recall:.3f}")
        for reranker_key, reranker_info in RERANKERS.items():
            if reranker_key not in results_by_reranker:
                results_by_reranker[reranker_key] = {}
                timing_data[reranker_key] = {}
            print(f"    Testing {reranker_key} ({reranker_info['name']})...")
            try:
                rerank_results, rerank_time = rerank_documents(
                    query=topic,
                    documents=documents,
                    reranker_info=reranker_info,
                    top_n=target_rerank_count,
                )
                timing_data[reranker_key][topic] = rerank_time
                reranked_indices = [
                    result["index"] for result in rerank_results["results"]
                ]
                reranked_points = [query_results.points[i] for i in reranked_indices]
            except Exception as e:
                print(f"      Reranking failed: {e}, using original top results")
                reranked_points = query_results.points[:target_rerank_count]
                timing_data[reranker_key][topic] = 0.0
            retrieved_topics = [point.payload["topic"] for point in reranked_points]
            true_positives = sum(1 for t in retrieved_topics if t == topic)
            total_retrieved = len(retrieved_topics)
            false_positives = total_retrieved - true_positives
            precision = true_positives / total_retrieved if total_retrieved > 0 else 0
            recall = (
                true_positives / total_relevant_docs if total_relevant_docs > 0 else 0
            )
            results_by_reranker[reranker_key][topic] = {
                "true_positives": true_positives,
                "false_positives": false_positives,
                "precision": precision,
                "recall": recall,
                "retrieved_topics": retrieved_topics,
                "total_retrieved": total_retrieved,
                "reranked_total": len(reranked_points),
                "original_total": original_total,
                "total_relevant_in_collection": total_relevant_docs,
            }
            print(f"      TP: {true_positives}, FP: {false_positives}")
            print(f"      Precision@{total_retrieved}: {precision:.3f}")
            print(f"      Recall@{total_retrieved}: {recall:.3f}")
            print(f"      Rerank Time: {rerank_time:.3f}s")
    return results_by_reranker, original_results, timing_data


def create_comprehensive_comparison_table(all_results, original_results, timing_data):
    summary_data = []
    orig_avg_precision = np.mean(
        [original_results[topic]["precision"] for topic in SPECIFIC_TOPICS]
    )
    orig_avg_recall = np.mean(
        [original_results[topic]["recall"] for topic in SPECIFIC_TOPICS]
    )
    orig_total_tp = sum(
        [original_results[topic]["true_positives"] for topic in SPECIFIC_TOPICS]
    )
    orig_total_retrieved = sum(
        [original_results[topic]["total_retrieved"] for topic in SPECIFIC_TOPICS]
    )
    orig_total_relevant = sum(
        [
            original_results[topic]["total_relevant_in_collection"]
            for topic in SPECIFIC_TOPICS
        ]
    )
    summary_data.append(
        {
            "Method": "ORIGINAL (No Reranking)",
            "Model_Name": "N/A",
            "Type": "baseline",
            "Avg_Precision": orig_avg_precision,
            "Avg_Recall": orig_avg_recall,
            "Precision_Improvement": 0.0,
            "Recall_Improvement": 0.0,
            "Avg_Rerank_Time": 0.0,
            "Total_TP": orig_total_tp,
            "Total_Retrieved": orig_total_retrieved,
            "Total_Relevant": orig_total_relevant,
            "Overall_Precision": orig_total_tp / orig_total_retrieved
            if orig_total_retrieved > 0
            else 0,
            "Overall_Recall": orig_total_tp / orig_total_relevant
            if orig_total_relevant > 0
            else 0,
        }
    )
    for reranker_key, reranker_info in RERANKERS.items():
        if reranker_key not in all_results:
            continue
        avg_precision = np.mean(
            [all_results[reranker_key][topic]["precision"] for topic in SPECIFIC_TOPICS]
        )
        avg_recall = np.mean(
            [all_results[reranker_key][topic]["recall"] for topic in SPECIFIC_TOPICS]
        )
        precision_improvement = avg_precision - orig_avg_precision
        recall_improvement = avg_recall - orig_avg_recall
        total_retrieved = sum(
            [
                all_results[reranker_key][topic]["total_retrieved"]
                for topic in SPECIFIC_TOPICS
            ]
        )
        total_tp = sum(
            [
                all_results[reranker_key][topic]["true_positives"]
                for topic in SPECIFIC_TOPICS
            ]
        )
        total_relevant = sum(
            [
                all_results[reranker_key][topic]["total_relevant_in_collection"]
                for topic in SPECIFIC_TOPICS
            ]
        )
        avg_time = np.mean(
            [timing_data[reranker_key][topic] for topic in SPECIFIC_TOPICS]
        )
        summary_data.append(
            {
                "Method": reranker_key.upper(),
                "Model_Name": reranker_info["name"],
                "Type": reranker_info["type"],
                "Avg_Precision": avg_precision,
                "Avg_Recall": avg_recall,
                "Precision_Improvement": precision_improvement,
                "Recall_Improvement": recall_improvement,
                "Avg_Rerank_Time": avg_time,
                "Total_TP": total_tp,
                "Total_Retrieved": total_retrieved,
                "Total_Relevant": total_relevant,
                "Overall_Precision": total_tp / total_retrieved
                if total_retrieved > 0
                else 0,
                "Overall_Recall": total_tp / total_relevant
                if total_relevant > 0
                else 0,
            }
        )
    return pd.DataFrame(summary_data)


async def main(k=100, score_threshold=0.3, rerank_top_percent=0.8):
    print("🔍 Evaluating Multiple Rerankers vs Original Results (CORRECTED)")
    print(
        f"Parameters: k={k}, threshold={score_threshold}, rerank_top={rerank_top_percent}"
    )
    print(f"Rerankers to test: {list(RERANKERS.keys())}")
    print("=" * 80)
    all_results = {}
    original_results = {}
    timing_data = {}
    for model_key, model_info in MODELS.items():
        print(f"\nEvaluating {model_info['name']} with multiple rerankers...")
        all_results, original_results, timing_data = await evaluate_rerankers(
            model_key, model_info, k, score_threshold, rerank_top_percent
        )
    print("\n🏆 Comprehensive Comparison Summary (CORRECTED METRICS):")
    summary_df = create_comprehensive_comparison_table(
        all_results, original_results, timing_data
    )
    print(summary_df.to_string(index=False, float_format="%.3f"))
    print("\n📊 Best Performing Methods:")
    if not summary_df.empty:
        best_precision = summary_df.loc[summary_df["Avg_Precision"].idxmax()]
        best_recall = summary_df.loc[summary_df["Avg_Recall"].idxmax()]
        best_precision_improvement = summary_df.loc[
            summary_df["Precision_Improvement"].idxmax()
        ]
        print(
            f"Best Precision: {best_precision['Method']} ({best_precision['Avg_Precision']:.3f})"
        )
        print(f"Best Recall: {best_recall['Method']} ({best_recall['Avg_Recall']:.3f})")
        print(
            f"Best Improvement: {best_precision_improvement['Method']} (+{best_precision_improvement['Precision_Improvement']:.3f})"
        )
    summary_df.to_csv("corrected_reranker_comparison.csv", index=False)
    print(
        f"\n💾 Corrected results saved to: {os.path.abspath('corrected_reranker_comparison.csv')}"
    )


if __name__ == "__main__":
    import sys

    k = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    score_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    rerank_percent = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    print(
        f"Using k={k}, score_threshold={score_threshold}, rerank_top_percent={rerank_percent}"
    )
    asyncio.run(main(k, score_threshold, rerank_percent))
