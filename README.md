# World Cup Social Media NLP Analysis

## Project Goal
This project explores social media discussions around the 2018 and 2022 FIFA World Cups using NLP techniques.

This project will focus on :

## comparative behavioral analysis using topics(topic modeling )
that answers those questions:
- Which topics dominated discussion during specific matches?
- Do fans from different countries focus on different themes?
- How topics evolved during the tournament?
- Are some topics niche or universal?
- Which topics generate the highest average likes?
- Which topics are most retweeted?
...



## Work Completed
  # Data & Setup
  - Collected datasets for FIFA World Cup 2018 & 2022 tweets
  - Initialized structured project architecture
  - Set up virtual environment for reproducibility
  # Data Exploration & Cleaning
   - Exploratory Data Analysis (EDA)
   - Text cleaning pipeline:
      - Removal of URLs, mentions, punctuation, and noise
      - Tokenization
      - Stopword removal
      - Lemmatization
  # Text Representation
  - Implemented TF-IDF vectorization
  - Converted tweets into numerical representations for modeling
  # Topic Modeling (NMF)
  - Applied Non-negative Matrix Factorization (NMF) for topic discovery
  - Extracted latent topics from tweet corpus

    * Observations:
      - Number of topics must be manually defined
      - Some topics contain noisy or unrelated words
      - Interpretation requires manual inspection
      - Topics are not always semantically coherent due to TF-IDF limitations
  # Topic Labeling
    - Applied KeyBERT for automatic keyword extraction
    - Generated initial topic labels based on representative keywords

      * Observations:
        - KeyBERT provides keywords but not fully semantic topic names
        - Manual interpretation is still required for final labeling
  

## Next Steps
  - Implement BERTopic for improved semantic topic modeling




