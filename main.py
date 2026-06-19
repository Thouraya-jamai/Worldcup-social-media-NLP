import os
import pickle

import pandas as pd

from src.NMF_pipeline import run_nmf_pipeline
from src.BERTopic_Modeling import run_bertopic_pipeline
# later you will add:
# from src.bertopic_pipeline import run_bertopic_pipeline


# ----------------------------
# CONFIG
# ----------------------------
DATASETS = [
    r"C:\Projects\worldcup-social-media-nlp\Data\Preprocessed_Data\worldcup2022.csv",
    r"C:\Projects\worldcup-social-media-nlp\Data\Preprocessed_Data\worldcup2018.csv"
]

N_TOPICS = 50

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)


def run_nmf_experiment(dataset_path):

    (
        df,
        vec,
        model,
        W,
        H,
        topic_labels
    ) = run_nmf_pipeline(
        dataset_path=dataset_path,
        n_topics=N_TOPICS
    )

    dataset_name = os.path.basename(dataset_path).replace(".csv", "")

    # Save dataset with topics
    df.to_csv(
        f"results/{dataset_name}_nmf_topics.csv",
        index=False
    )
     # SAVE TOPIC LABELS
    rows = []
    for topic_id, keywords in topic_labels.items():
        rows.append({
        "topic_id": topic_id,
        "keybert_labels": keywords})
    topic_df = pd.DataFrame(rows)
    
    topic_df.to_csv(
    f"results/{dataset_name}_topic_labels.csv",
    index=False)
    

    # Save models
    pickle.dump(
        vec,
        open(f"models/{dataset_name}_vectorizer.pkl", "wb")
    )

    pickle.dump(
        model,
        open(f"models/{dataset_name}_nmf.pkl", "wb")
    )

    print(f"\n[NMF DONE] Saved results for {dataset_name}")

def run_bertopic_experiment(dataset_path):

    (
        df,
        topic_model,
        topics,
        probs,
        topic_info
    ) = run_bertopic_pipeline(dataset_path)

    dataset_name = os.path.basename(dataset_path).replace(".csv", "")

    # Save tweet-level results
    df.to_csv(
        f"results/{dataset_name}_bertopic_topics.csv",
        index=False
    )

    # Save topic summary
    topic_info.to_csv(
        f"results/{dataset_name}_bertopic_topic_info.csv",
        index=False
    )

    # Save model
    topic_model.save(
        f"models/{dataset_name}_bertopic"
    )

    print(f"\n[BERTopic DONE] {dataset_name}")
# ----------------------------
# RUN ALL DATASETS
# ----------------------------
if __name__ == "__main__":

    MODE = "both"   # options: "nmf", "bertopic", "both"

    for dataset_path in DATASETS:

        print("\n" + "="*80)
        print(f"Dataset: {dataset_path}")

        if MODE in ["bertopic", "both"]:
            print("\n--- Running BERTopic ---")
            run_bertopic_experiment(dataset_path)

        if MODE in ["nmf", "both"]:
            print("\n--- Running NMF ---")
            run_nmf_experiment(dataset_path)