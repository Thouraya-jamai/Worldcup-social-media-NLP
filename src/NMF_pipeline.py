import pandas as pd
from src.nmf_modeling import train_nmf
from src.Text_representation import create_tdidf_matrix
from src.KeybertModeling import label_topics_with_keybert

def run_nmf_pipeline(dataset_path,n_topics,text_column="Tweet"):
    
    # Load dataset
    df = pd.read_csv(dataset_path)

    texts = df[text_column].fillna("")

    # TF-IDF
    X, vectorizer = create_tdidf_matrix(texts)

    # NMF
    nmf_model, W, H = train_nmf(
        X,
        n_topics=n_topics
    )

    # Dominant topic per tweet
    df["dominant_topic"] = W.argmax(axis=1)

    # KeyBERT labeling
    topic_labels = label_topics_with_keybert(df)

    return (
        df,
        vectorizer,
        nmf_model,
        W,
        H,
        topic_labels
    )