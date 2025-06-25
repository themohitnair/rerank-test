import asyncio
from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

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

SPECIFIC_TOPICS = [
    "England Cricket Team",
    "England Football Team",
    "Black Coffee",
    "Indian Cricket Team",
    "Black History Month",
]

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

async def evaluate_model(model_key, model_info, local_models=None, k=100, score_threshold=0.3):
    results = {}
    collection_name = f"{model_key}_posts"

    for topic in SPECIFIC_TOPICS:
        print(f"Querying '{topic}' in {collection_name}...")

        if model_info["type"] == "local":
            query_vector = local_models[model_key].encode(topic).tolist()
        elif model_info["type"] == "openai_api":
            query_vector = get_openai_embedding(topic, model_info["name"])
        elif model_info["type"] == "jina_api":
            query_vector = get_jina_embedding(topic, model_info["name"])

        query_results = await client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
            score_threshold=score_threshold,
        )

        retrieved_topics = [point.payload["topic"] for point in query_results.points]

        true_positives = sum(1 for t in retrieved_topics if t == topic)
        total_retrieved = len(retrieved_topics)
        false_positives = total_retrieved - true_positives

        precision = true_positives / total_retrieved if total_retrieved > 0 else 0

        results[topic] = {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "precision": precision,
            "recall": true_positives / 100,
            "retrieved_topics": retrieved_topics,
            "total_retrieved": total_retrieved,
        }

        print(f"  Retrieved: {total_retrieved}, TP: {true_positives}, FP: {false_positives}")
        print(f"  Precision: {precision:.3f}, Recall: {true_positives / 100:.3f}")

    return results

def plot_separate_topics(all_results):
    import os
    os.makedirs("topic_plots", exist_ok=True)

    for topic in SPECIFIC_TOPICS:
        fig, axes = plt.subplots(1, len(MODELS), figsize=(25, 5), squeeze=False)
        fig.suptitle(f'Topic Distribution for "{topic}" Across All Models', fontsize=16)

        for idx, (model_key, model_info) in enumerate(MODELS.items()):
            counter = Counter(all_results[model_key][topic]["retrieved_topics"])
            labels = list(counter.keys())
            counts = list(counter.values())

            ax = axes[0, idx] if len(MODELS) > 1 else axes[0, 0]

            bars = ax.bar(range(len(labels)), counts, color=model_info["color"])
            
            for i, label in enumerate(labels):
                if label == topic:
                    bars[i].set_color('gold')
                    bars[i].set_edgecolor('black')
                    bars[i].set_linewidth(2)

            ax.set_title(
                f"{model_key.upper()}\n(P:{all_results[model_key][topic]['precision']:.3f}, R:{all_results[model_key][topic]['recall']:.3f})",
                fontsize=10,
            )
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Count")

            ax.text(
                0.02, 0.98,
                f"Total: {all_results[model_key][topic]['total_retrieved']}",
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        safe_topic_name = topic.replace(" ", "_").replace("/", "_")
        plt.savefig(f"topic_plots/{safe_topic_name}_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved plot for '{topic}'")

def plot_performance_comparison(all_results):
    topics = SPECIFIC_TOPICS
    models = list(MODELS.keys())

    precision_data = []
    recall_data = []

    for model_key in models:
        model_precisions = [all_results[model_key][topic]["precision"] for topic in topics]
        model_recalls = [all_results[model_key][topic]["recall"] for topic in topics]
        precision_data.append(model_precisions)
        recall_data.append(model_recalls)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    x = np.arange(len(topics))
    width = 0.08
    for i, (model_key, model_info) in enumerate(MODELS.items()):
        offset = (i - len(MODELS) / 2) * width
        ax1.bar(x + offset, precision_data[i], width, label=model_key.upper(), color=model_info["color"])
        ax2.bar(x + offset, recall_data[i], width, label=model_key.upper(), color=model_info["color"])

    ax1.set_title("Precision Comparison Across Topics")
    ax1.set_xlabel("Topics")
    ax1.set_ylabel("Precision")
    ax1.set_xticks(x)
    ax1.set_xticklabels(topics, rotation=45, ha="right")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Recall Comparison Across Topics")
    ax2.set_xlabel("Topics")
    ax2.set_ylabel("Recall")
    ax2.set_xticks(x)
    ax2.set_xticklabels(topics, rotation=45, ha="right")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Saved performance comparison plot")

def plot_all_models_comparison(all_results):
    fig, axes = plt.subplots(len(MODELS), len(SPECIFIC_TOPICS), figsize=(30, 4 * len(MODELS)), squeeze=False)
    fig.suptitle("Topic Distribution Comparison Across All Models", fontsize=16)

    for row, (model_key, model_info) in enumerate(MODELS.items()):
        for col, topic in enumerate(SPECIFIC_TOPICS):
            counter = Counter(all_results[model_key][topic]["retrieved_topics"])
            labels = list(counter.keys())
            counts = list(counter.values())

            ax = axes[row, col]
            bars = ax.bar(range(len(labels)), counts, color=model_info["color"])
            
            for i, label in enumerate(labels):
                if label == topic:
                    bars[i].set_color('gold')
                    bars[i].set_edgecolor('black')
                    bars[i].set_linewidth(2)
            
            ax.set_title(f"{topic}\n({model_key.upper()})", fontsize=9)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("all_models_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Saved all models comparison plot")

def create_comprehensive_table(all_results):
    comparison_data = []

    for topic in SPECIFIC_TOPICS:
        row = {"Topic": topic}

        for model_key in MODELS.keys():
            row[f"{model_key.upper()}_Retrieved"] = all_results[model_key][topic]["total_retrieved"]
            row[f"{model_key.upper()}_TP"] = all_results[model_key][topic]["true_positives"]
            row[f"{model_key.upper()}_FP"] = all_results[model_key][topic]["false_positives"]
            row[f"{model_key.upper()}_Precision"] = all_results[model_key][topic]["precision"]
            row[f"{model_key.upper()}_Recall"] = all_results[model_key][topic]["recall"]

        comparison_data.append(row)

    return pd.DataFrame(comparison_data)

def create_summary_table(all_results):
    summary_data = []

    for model_key, model_info in MODELS.items():
        avg_precision = np.mean([all_results[model_key][topic]["precision"] for topic in SPECIFIC_TOPICS])
        avg_recall = np.mean([all_results[model_key][topic]["recall"] for topic in SPECIFIC_TOPICS])
        total_retrieved = sum([all_results[model_key][topic]["total_retrieved"] for topic in SPECIFIC_TOPICS])
        total_tp = sum([all_results[model_key][topic]["true_positives"] for topic in SPECIFIC_TOPICS])
        total_fp = sum([all_results[model_key][topic]["false_positives"] for topic in SPECIFIC_TOPICS])

        summary_data.append({
            "Model": model_key.upper(),
            "Model_Name": model_info["name"],
            "Dimensions": model_info["dim"],
            "Type": model_info["type"],
            "Avg_Precision": avg_precision,
            "Avg_Recall": avg_recall,
            "Total_Retrieved": total_retrieved,
            "Total_TP": total_tp,
            "Total_FP": total_fp,
            "Overall_Precision": total_tp / total_retrieved if total_retrieved > 0 else 0,
        })

    return pd.DataFrame(summary_data).sort_values("Avg_Precision", ascending=False)

async def main(k=100, score_threshold=0.3):
    print(f"🔍 Evaluating {len(MODELS)} embedding models with k={k}, threshold={score_threshold}")
    print("=" * 80)

    local_models = {}
    all_results = {}

    for model_key, model_info in MODELS.items():
        if model_info["type"] == "local":
            print(f"Loading {model_info['name']}...")
            local_models[model_key] = SentenceTransformer(model_info["name"])

    for model_key, model_info in MODELS.items():
        print(f"\nEvaluating {model_info['name']}...")
        all_results[model_key] = await evaluate_model(model_key, model_info, local_models, k, score_threshold)

    print("\n📊 Generating comprehensive comparison charts...")
    plot_all_models_comparison(all_results)

    print("\n📊 Generating separate topic plots...")
    plot_separate_topics(all_results)

    print("\n📊 Generating performance comparison plots...")
    plot_performance_comparison(all_results)

    print("\n📋 Detailed Results:")
    detailed_df = create_comprehensive_table(all_results)
    print(detailed_df.to_string(index=False, float_format="%.3f"))

    print("\n🏆 Model Performance Summary:")
    summary_df = create_summary_table(all_results)
    print(summary_df.to_string(index=False, float_format="%.3f"))

    print(f"\n🥇 Winner: {summary_df.iloc[0]['Model']} ({summary_df.iloc[0]['Model_Name']})")
    print(f"   Avg Precision: {summary_df.iloc[0]['Avg_Precision']:.3f}")
    print(f"   Dimensions: {summary_df.iloc[0]['Dimensions']}")
    print(f"   Type: {summary_df.iloc[0]['Type']}")

    detailed_df.to_csv("detailed_model_comparison.csv", index=False)
    summary_df.to_csv("model_summary.csv", index=False)
    print("\n💾 Results saved to CSV files and various visualization plots")
    print("📁 Individual topic plots saved in 'topic_plots/' directory")

if __name__ == "__main__":
    import sys
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    score_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    print(f"Using k={k}, score_threshold={score_threshold}")
    asyncio.run(main(k, score_threshold))
