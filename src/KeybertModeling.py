from keybert import KeyBERT

kw_model = KeyBERT()

def label_topics_with_keybert(df, n_tweets=50, top_n=5):
    
    topic_labels = {}

    for topic_id in df["dominant_topic"].unique():

        # Combine tweets of this topic
        topic_text = " ".join(
            df[df["dominant_topic"] == topic_id]["Tweet"]
            .dropna()
            .astype(str)
            .head(n_tweets)
        )

        # Extract keywords
        keywords = kw_model.extract_keywords(
            topic_text,
            top_n=top_n
        )

        topic_labels[topic_id] = keywords

    return topic_labels