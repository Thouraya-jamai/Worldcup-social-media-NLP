import re
import nltk
import string
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#clean noise ( punctuations, URLs, mentions and extra spaces)
def clean_noise(text):
    text = re.sub(r"http\S+", " ", text)      # remove URLs
    text = re.sub(r"@\w+", " ", text)         # remove mentions
    text = re.sub(r"[^\w\s]", " ", text)      # remove punctuation
    text = re.sub(r"\s+", " ", text)          # remove extra spaces     
    return text.strip()

#tokenization
def tokenization(text):
    return word_tokenize(text)

#stop_words removal
stop_words=set(stopwords.words('english'))
def stopwords_removal(tokens):
    return[token for token in tokens if token not in stop_words ]

#text normalization
lemmatizer=WordNetLemmatizer()
def lemmatization(tokens):
    #return[lemmatizer.lemmatize(token for token in tokens)]
    return [lemmatizer.lemmatize(token) for token in tokens]
    
#the whole pipeline 
def preprocess_text(text):
    text=clean_noise(text)
    tokens=tokenization(text)
    tokens=stopwords_removal(tokens)
    tokens=lemmatization(tokens)
    return tokens