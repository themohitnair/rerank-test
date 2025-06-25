import asyncio
from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
import matplotlib.pyplot as plt
from collections import Counter
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

MODELS = {
    "e5_large": {
        "name": "intfloat/e5-large-v2",
        "dim": 1024,
        "color": "lightgreen",
        "type": "local",
    },
    "e5_base": {
        "name": "intfloat/e5-base-v2",
        "dim": 768,
        "color": "lightyellow",
        "type": "local",
    },
}

RERANKERS = {
    "mxbai_large": {
        "name": "mixedbread-ai/mxbai-rerank-large-v1",
        "type": "sentence_transformer",
        "color": "orange",
    },
    "bge_large": {
        "name": "BAAI/bge-reranker-large",
        "type": "flag_embedding",
        "color": "darkgreen",
    },
    "bge_v2_m3": {
        "name": "BAAI/bge-reranker-v2-m3",
        "type": "sentence_transformer",
        "color": "blue",
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
    scroll_result = await client.scroll(collection_name=collection_name, limit=10000, with_payload=True)
    all_points = scroll_result[0]
    for point in all_points:
        topic = point.payload.get("topic", "Unknown")
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    return topic_counts

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
        elif reranker_type == "flag_embedding":
            result = flag_rerank(query, documents, model_name, top_n)
        elif reranker_type == "sentence_transformer":
            result = sentence_transformer_rerank(query, documents, model_name, top_n)
        else:
            raise ValueError(f"Unknown reranker type: {reranker_type}")
    except Exception as e:
        print(f"Reranking failed for {model_name}: {e}")
        result = {"results": [{"index": i, "relevance_score": 0.0} for i in range(min(top_n or len(documents), len(documents)))]}
    end_time = time.time()
    rerank_time = end_time - start_time
    return result, rerank_time

async def evaluate_rerankers(model_key, model_info, local_models, k=100, score_threshold=0.3, rerank_top_percent=0.8):
    results_by_reranker = {}
    original_results = {}
    timing_data = {}
    collection_name = f"{model_key}_posts"
    topic_counts = await get_topic_counts(collection_name)
    
    for topic in SPECIFIC_TOPICS:
        print(f"Querying '{topic}' in {collection_name}...")
        total_relevant_docs = topic_counts.get(topic, 0)
        
        if model_info["type"] == "local":
            query_vector = local_models[model_key].encode(topic).tolist()
        else:
            raise ValueError(f"Unknown model type: {model_info['type']}")
        
        query_results = await client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
            score_threshold=score_threshold,
        )
        
        if not query_results.points:
            continue
            
        original_total = len(query_results.points)
        target_rerank_count = int(len(query_results.points) * rerank_top_percent)
        original_topics = [point.payload["topic"] for point in query_results.points[:target_rerank_count]]
        original_tp = sum(1 for t in original_topics if t == topic)
        original_precision = original_tp / len(original_topics) if len(original_topics) > 0 else 0
        original_recall = original_tp / total_relevant_docs if total_relevant_docs > 0 else 0
        
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
        
        for reranker_key, reranker_info in RERANKERS.items():
            if reranker_key not in results_by_reranker:
                results_by_reranker[reranker_key] = {}
                timing_data[reranker_key] = {}
            
            try:
                rerank_results, rerank_time = rerank_documents(
                    query=topic, documents=documents, reranker_info=reranker_info, top_n=target_rerank_count
                )
                timing_data[reranker_key][topic] = rerank_time
                reranked_indices = [result["index"] for result in rerank_results["results"]]
                reranked_points = [query_results.points[i] for i in reranked_indices]
            except Exception as e:
                print(f"      Reranking failed: {e}")
                reranked_points = query_results.points[:target_rerank_count]
                timing_data[reranker_key][topic] = 0.0
            
            retrieved_topics = [point.payload["topic"] for point in reranked_points]
            true_positives = sum(1 for t in retrieved_topics if t == topic)
            total_retrieved = len(retrieved_topics)
            false_positives = total_retrieved - true_positives
            precision = true_positives / total_retrieved if total_retrieved > 0 else 0
            recall = true_positives / total_relevant_docs if total_relevant_docs > 0 else 0
            
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
    
    return results_by_reranker, original_results, timing_data

def plot_reranker_performance_comparison(all_results, original_results, timing_data, model_name):
    """Plot comprehensive reranker performance analysis"""
    
    # 1. Precision/Recall Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'Reranker Performance Analysis for {model_name}', fontsize=16)
    
    topics = SPECIFIC_TOPICS
    x = np.arange(len(topics))
    width = 0.15
    
    # Precision comparison
    original_precisions = [original_results[topic]["precision"] for topic in topics]
    ax1.bar(x - width*1.5, original_precisions, width, label='Original', color='gray', alpha=0.7)
    
    for i, (reranker_key, reranker_info) in enumerate(RERANKERS.items()):
        if reranker_key in all_results:
            precisions = [all_results[reranker_key][topic]["precision"] for topic in topics]
            ax1.bar(x + width*(i-0.5), precisions, width, label=reranker_key.upper(), color=reranker_info["color"])
    
    ax1.set_title("Precision Comparison: Original vs Rerankers")
    ax1.set_xlabel("Topics")
    ax1.set_ylabel("Precision")
    ax1.set_xticks(x)
    ax1.set_xticklabels(topics, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Recall comparison  
    original_recalls = [original_results[topic]["recall"] for topic in topics]
    ax2.bar(x - width*1.5, original_recalls, width, label='Original', color='gray', alpha=0.7)
    
    for i, (reranker_key, reranker_info) in enumerate(RERANKERS.items()):
        if reranker_key in all_results:
            recalls = [all_results[reranker_key][topic]["recall"] for topic in topics]
            ax2.bar(x + width*(i-0.5), recalls, width, label=reranker_key.upper(), color=reranker_info["color"])
    
    ax2.set_title("Recall Comparison: Original vs Rerankers")
    ax2.set_xlabel("Topics")
    ax2.set_ylabel("Recall")
    ax2.set_xticks(x)
    ax2.set_xticklabels(topics, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Average precision improvement
    avg_precisions = []
    reranker_names = ['Original']
    avg_precisions.append(np.mean(original_precisions))
    
    for reranker_key, reranker_info in RERANKERS.items():
        if reranker_key in all_results:
            avg_prec = np.mean([all_results[reranker_key][topic]["precision"] for topic in topics])
            avg_precisions.append(avg_prec)
            reranker_names.append(reranker_key.upper())
    
    bars3 = ax3.bar(reranker_names, avg_precisions, color=['gray'] + [RERANKERS[k]["color"] for k in RERANKERS.keys() if k in all_results])
    ax3.set_title("Average Precision by Method")
    ax3.set_ylabel("Average Precision")
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars3, avg_precisions):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Timing analysis
    avg_times = []
    timing_names = []
    timing_colors = []
    
    for reranker_key, reranker_info in RERANKERS.items():
        if reranker_key in timing_data:
            avg_time = np.mean([timing_data[reranker_key][topic] for topic in topics])
            avg_times.append(avg_time)
            timing_names.append(reranker_key.upper())
            timing_colors.append(reranker_info["color"])
    
    bars4 = ax4.bar(timing_names, avg_times, color=timing_colors)
    ax4.set_title("Average Reranking Time")
    ax4.set_ylabel("Time (seconds)")
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, time_val in zip(bars4, avg_times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{time_val:.3f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    safe_model_name = model_name.replace("/", "_").replace(" ", "_")
    plt.savefig(f"reranker_performance_{safe_model_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved reranker performance plot: reranker_performance_{safe_model_name}.png")

def plot_topic_distribution_comparison(all_results, original_results, model_name):
    """Plot topic distribution for original vs reranked results"""
    
    os.makedirs("reranker_topic_plots", exist_ok=True)
    
    for topic in SPECIFIC_TOPICS:
        fig, axes = plt.subplots(1, len(RERANKERS) + 1, figsize=(6 * (len(RERANKERS) + 1), 6))
        fig.suptitle(f'Topic Distribution: "{topic}" - Original vs Rerankers ({model_name})', fontsize=16)
        
        # Plot original results
        counter = Counter(original_results[topic]["retrieved_topics"])
        labels = list(counter.keys())
        counts = list(counter.values())
        
        ax = axes[0]
        bars = ax.bar(range(len(labels)), counts, color='gray', alpha=0.7)
        
        # Highlight target topic
        for i, label in enumerate(labels):
            if label == topic:
                bars[i].set_color('gold')
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(2)
        
        ax.set_title(f"ORIGINAL\nP:{original_results[topic]['precision']:.3f}")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Count")
        
        # Plot reranked results
        for idx, (reranker_key, reranker_info) in enumerate(RERANKERS.items(), 1):
            if reranker_key in all_results and topic in all_results[reranker_key]:
                counter = Counter(all_results[reranker_key][topic]["retrieved_topics"])
                labels = list(counter.keys())
                counts = list(counter.values())
                
                ax = axes[idx]
                bars = ax.bar(range(len(labels)), counts, color=reranker_info["color"], alpha=0.7)
                
                # Highlight target topic
                for i, label in enumerate(labels):
                    if label == topic:
                        bars[i].set_color('gold')
                        bars[i].set_edgecolor('black')
                        bars[i].set_linewidth(2)
                
                precision_diff = all_results[reranker_key][topic]['precision'] - original_results[topic]['precision']
                ax.set_title(f"{reranker_key.upper()}\nP:{all_results[reranker_key][topic]['precision']:.3f} (Δ{precision_diff:+.3f})")
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                ax.set_ylabel("Count")
        
        plt.tight_layout()
        safe_topic_name = topic.replace(" ", "_").replace("/", "_")
        safe_model_name = model_name.replace("/", "_").replace(" ", "_")
        plt.savefig(f"reranker_topic_plots/{safe_topic_name}_{safe_model_name}_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved topic distribution plot: {safe_topic_name}_{safe_model_name}_comparison.png")

def create_comprehensive_comparison_table(all_results, original_results, timing_data):
    summary_data = []
    
    # Original results
    orig_avg_precision = np.mean([original_results[topic]["precision"] for topic in SPECIFIC_TOPICS])
    orig_avg_recall = np.mean([original_results[topic]["recall"] for topic in SPECIFIC_TOPICS])
    orig_total_tp = sum([original_results[topic]["true_positives"] for topic in SPECIFIC_TOPICS])
    orig_total_retrieved = sum([original_results[topic]["total_retrieved"] for topic in SPECIFIC_TOPICS])
    orig_total_relevant = sum([original_results[topic]["total_relevant_in_collection"] for topic in SPECIFIC_TOPICS])
    
    summary_data.append({
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
        "Overall_Precision": orig_total_tp / orig_total_retrieved if orig_total_retrieved > 0 else 0,
        "Overall_Recall": orig_total_tp / orig_total_relevant if orig_total_relevant > 0 else 0,
    })
    
    # Reranker results
    for reranker_key, reranker_info in RERANKERS.items():
        if reranker_key not in all_results:
            continue
            
        avg_precision = np.mean([all_results[reranker_key][topic]["precision"] for topic in SPECIFIC_TOPICS])
        avg_recall = np.mean([all_results[reranker_key][topic]["recall"] for topic in SPECIFIC_TOPICS])
        precision_improvement = avg_precision - orig_avg_precision
        recall_improvement = avg_recall - orig_avg_recall
        
        total_retrieved = sum([all_results[reranker_key][topic]["total_retrieved"] for topic in SPECIFIC_TOPICS])
        total_tp = sum([all_results[reranker_key][topic]["true_positives"] for topic in SPECIFIC_TOPICS])
        total_relevant = sum([all_results[reranker_key][topic]["total_relevant_in_collection"] for topic in SPECIFIC_TOPICS])
        
        avg_time = np.mean([timing_data[reranker_key][topic] for topic in SPECIFIC_TOPICS])
        
        summary_data.append({
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
            "Overall_Precision": total_tp / total_retrieved if total_retrieved > 0 else 0,
            "Overall_Recall": total_tp / total_relevant if total_relevant > 0 else 0,
        })
    
    return pd.DataFrame(summary_data)

async def main(k=100, score_threshold=0.3, rerank_top_percent=0.8):
    print("🔍 Evaluating Multiple Rerankers with Local E5 Embeddings")
    print(f"Parameters: k={k}, threshold={score_threshold}, rerank_top={rerank_top_percent}")
    print(f"Embedding models: {list(MODELS.keys())}")
    print(f"Rerankers to test: {list(RERANKERS.keys())}")
    print("=" * 80)
    
    # Load local embedding models
    local_models = {}
    for model_key, model_info in MODELS.items():
        if model_info["type"] == "local":
            print(f"Loading {model_info['name']}...")
            local_models[model_key] = SentenceTransformer(model_info["name"])
    
    # Store results for all models
    all_model_results = {}
    
    for model_key, model_info in MODELS.items():
        print(f"\nEvaluating {model_info['name']} with multiple rerankers...")
        all_results, original_results, timing_data = await evaluate_rerankers(
            model_key, model_info, local_models, k, score_threshold, rerank_top_percent
        )
        
        # Store results for this model
        all_model_results[model_key] = {
            'all_results': all_results,
            'original_results': original_results, 
            'timing_data': timing_data,
            'model_info': model_info
        }
        
        print(f"\n📊 Generating performance plots for {model_info['name']}...")
        plot_reranker_performance_comparison(all_results, original_results, timing_data, model_info['name'])
        
        print(f"\n📊 Generating topic distribution plots for {model_info['name']}...")
        plot_topic_distribution_comparison(all_results, original_results, model_info['name'])
        
        print(f"\n🏆 Results Summary for {model_info['name']}:")
        summary_df = create_comprehensive_comparison_table(all_results, original_results, timing_data)
        print(summary_df.to_string(index=False, float_format="%.3f"))
        
        # Save results for this model
        model_filename = f"reranker_results_{model_key}.csv"
        summary_df.to_csv(model_filename, index=False)
        print(f"💾 Results saved to: {os.path.abspath(model_filename)}")
        
        if not summary_df.empty:
            best_precision = summary_df.loc[summary_df["Avg_Precision"].idxmax()]
            best_improvement = summary_df.loc[summary_df["Precision_Improvement"].idxmax()]
            fastest = summary_df[summary_df["Method"] != "ORIGINAL (No Reranking)"]
            if not fastest.empty:
                fastest_reranker = fastest.loc[fastest["Avg_Rerank_Time"].idxmin()]
                print(f"\n📊 Best Results for {model_info['name']}:")
                print(f"   🎯 Best Precision: {best_precision['Method']} ({best_precision['Avg_Precision']:.3f})")
                print(f"   📈 Best Improvement: {best_improvement['Method']} (+{best_improvement['Precision_Improvement']:.3f})")
                print(f"   ⚡ Fastest: {fastest_reranker['Method']} ({fastest_reranker['Avg_Rerank_Time']:.3f}s)")

if __name__ == "__main__":
    import sys
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    score_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    rerank_percent = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    print(f"Using k={k}, score_threshold={score_threshold}, rerank_top_percent={rerank_percent}")
    asyncio.run(main(k, score_threshold, rerank_percent))
