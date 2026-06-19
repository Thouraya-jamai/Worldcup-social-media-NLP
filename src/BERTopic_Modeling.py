import pandas as pd
from bertopic import BERTopic


def run_bertopic_pipeline(dataset_path, text_column="Tweet"):

    # Load data
    df = pd.read_csv(dataset_path)

    # Clean input
    tweets = df[text_column].fillna("").astype(str).tolist()

    # Model
    topic_model = BERTopic(verbose=True)

    # Fit + transform
    topics, probs = topic_model.fit_transform(tweets)

    # Save topics in dataframe
    df["topic"] = topics
    topic_info = topic_model.get_topic_info()

    return df, topic_model, topics, probs,topic_info

