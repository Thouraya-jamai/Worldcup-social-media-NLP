from sklearn.decomposition import NMF

def train_nmf(X, n_topics=50, random_state=42):
    model = NMF(n_components=n_topics, random_state=random_state)
    
    W = model.fit_transform(X)   # document-topic
    H = model.components_        # topic-word
    
    return model, W, H


   
