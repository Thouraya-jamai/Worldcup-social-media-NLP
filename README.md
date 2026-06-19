# World Cup Social Media NLP Analysis

## Project Goal

This project explores social media discussions surrounding the FIFA World Cups of 2018 and 2022 using Natural Language Processing (NLP) techniques.

The objective is to identify and compare fan behaviors, discussion themes, engagement patterns, and sentiment across tournaments through topic modeling and social media analytics.

---

## Research Questions

### Topic Modeling & Behavioral Analysis

* Which topics dominated discussion during the tournaments?
* Do fans from different countries focus on different themes?
* Which topics are universal and which are niche?
* Which topics receive the highest average number of likes?
* Which topics receive the highest average number of retweets?
* Which topics are associated with positive, negative, or neutral sentiment?
* Which topics are the most controversial?
* Which locations generate the highest volume of World Cup discussions?
* What hashtags are most frequently used and where are they most popular?

---

## Datasets

### FIFA World Cup 2018 Dataset

Features:

* ID       
* lang       
* Date
* Source
* len
* Orig_Tweet
* Tweet
* Likes
* RTs
* Hashtags
* UserMentionNames
* Name
* Place
* Followers
* Friends

### FIFA World Cup 2022 Dataset

Features:

* Unnamed: 0
* Date Created
* Source of Tweet
* Tweet
* Number of Likes
* Sentiment

---

## Work Completed

### Data Preparation

* Structured project architecture
* Virtual environment setup
* Data loading and preprocessing pipelines

### Exploratory Data Analysis (EDA)

* Dataset exploration
* Missing value analysis
* Distribution analysis
* Location and hashtag exploration

### Text Preprocessing

* URL removal
* Mention removal
* Noise and punctuation cleaning
* Tokenization
* Stopword removal
* Lemmatization

### Text Representation

* TF-IDF Vectorization
* Document-Term Matrix construction

---

## Topic Modeling

### Non-negative Matrix Factorization (NMF)

* Trained NMF models on both datasets
* Extracted latent discussion topics
* Generated dominant topic assignments for each tweet

#### Advantages

* Fast and interpretable
* Works well with TF-IDF features

#### Limitations

* Number of topics must be predefined
* Topic interpretation requires manual labeling
* Some topics may contain noisy or overlapping keywords

---

### BERTopic

* Applied transformer-based topic modeling
* Generated semantic topic representations
* Extracted representative documents
* Identified outlier topics automatically

#### Advantages

* More coherent and interpretable topics
* Automatic topic reduction
* Representative tweets facilitate topic labeling

#### Limitations

* Computationally more expensive
* Still requires manual topic naming

---

## Topic Labeling

* Applied KeyBERT for keyword extraction
* Used topic representations and representative tweets for manual interpretation
* Assigned semantic labels to major topics

---

## Behavioral Analysis

### 2018 World Cup

* Most discussed topics
* Topic distribution by location
* Most active locations
* Most liked topics
* Most retweeted topics
* Universal vs niche topics
* Hashtag usage across locations
* Topic evolution during the tournament

### 2022 World Cup

* Topic distribution
* Sentiment distribution by topic
* Top positive topics
* Top negative topics
* Most liked topics
* Most controversial topics
* Topic evolution over time
* Source analysis

---

## Tools & Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* NLTK
* BERTopic
* Sentence-Transformers
* UMAP
* HDBSCAN
* KeyBERT

---

## Project Outcome

The project provides a comparative behavioral analysis of FIFA World Cup discussions across social media by combining topic modeling, sentiment analysis, engagement metrics, and geographical patterns. Results highlight how fan interests, emotions, and discussion themes evolved between the 2018 and 2022 tournaments.
