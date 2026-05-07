#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
def create_tdidf_matrix(text_series, max_features=5000,ngram_range=(1,2)):
    text_series = text_series.fillna('').astype(str)
    vectorizer=TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X=vectorizer.fit_transform(text_series)
    return X,vectorizer


