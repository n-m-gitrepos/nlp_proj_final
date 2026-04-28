# nlp_proj_final

# MS MARCO Passage Ranking with Sentence-BERT

This project implements a passage ranking system using the MS MARCO dataset and Sentence-BERT. The goal is to train a model that can determine how relevant a passage is to a given query, and rank passages accordingly.

Project Pipeline:

1. Data preprocessing
2. Training a baseline Sentence-BERT model
3. Enhancing model with lexical features
4. Evaluation, comparing performance, and analysis

---

## Dataset Information

We use part of the MS MARCO Passage Ranking dataset, which consists of:

* **collection.tsv**: Contains passages: chunks of text extrcted from Bing Search Engine used as context for answering queries.

  * Format: `passage_id \t passage_text`

* **queries.train.tsv / queries.dev.tsv**: Contains query sets: real user queries associated with a passage.

  * Format: `query_id \t query_text`

* **qrels.train.tsv / qrels.dev.tsv**: Human verified ground truth labels - a subset of collection.tsv.

  * Format: `query_id \t 0 \t passage_id \t relevance`

### Additional Ground Truth Information

* Passages in qrels → **relevant (label = 1)**
* All other passages → **non-relevant (label = 0)**

---

## Pipeline Overview

```
Raw TSV Data
     ↓
Preprocessing (preprocess.py)
     ↓
Processed JSON Data
     ↓
Data Loading (data_loader.py)
     ↓
Model Training (train.py)
     ↓
Ranking + Evaluation (evaluate.py)
```

---

## Step 1: Data Preprocessing

File: `preprocessing.py`

### Goals

* Transform raw data into a format ready for training
* Generate labeled query-passage pairs
* Create evaluation (dev) candidate sets

### Key Steps

#### 1. Load Data

* Map queries + passages to their respective ids:

  * `passage_id → passage_text`
  * `query_id → query_text`

#### 2. Parse Ground Truth (qrels)

* Map:

  * `query_id → set of relevant passage_ids`

#### 3. Query Sampling

* Select a subset of queries (e.g., 100) to train on
* Keeps training computationally manageable

#### 4. Negative Sampling

* For each query:

  * Include all relevant passages (positives)
  * Sample non-relevant passages (negatives)

#### 5. Create Training Data

Format:

```
(query, passage, label)
```

#### 6. Create Evaluation Data

For each query:

* Include all positives
* Add random negatives to form candidate pool (~100)

### Outputs

* `train_pairs.json`
* `dev_candidates.json`

---

## Step 2: Data Loading

File: `data_loader.py`

### Purpose

Convert JSON data into Sentence-BERT compatible format.

### Output Format

Each example becomes:

```
InputExample(
  texts=[query, passage],
  label=0.0 or 1.0
)
```

---

## Step 3: Model Training

File: `train.py`

### Model

* Sentence-BERT (SBERT)
* Produces embeddings for queries and passages

### Training Objective

* Relevant pairs → high similarity
* Non-relevant pairs → low similarity

### Common Loss Functions

* CosineSimilarityLoss
* MultipleNegativesRankingLoss (advanced)

---

## Step 4: Ranking and Scoring

After training, the model is used to compute relevance scores.

### Process

1. Encode query into embedding
2. Encode candidate passages
3. Compute cosine similarity

### Score Range

* Raw: [-1, 1]
* Normalized: [0, 1] or [0, 100]

### Ranking

* Sort passages by similarity score
* Select top-k (e.g., top 5–10)

---

## Step 5: Evaluation

File: `evaluate.py`

### Metrics

#### 1. Recall@K

* Measures how many relevant passages appear in top-K

#### 2. Mean Reciprocal Rank (MRR)

* Measures how early the first relevant result appears

#### 3. Accuracy@K

* Checks if any relevant result appears in top-K

#### 4. NDCG (optional)

* Accounts for ranking quality

---

## Step 6: Model Improvement (Lexical Features)

To improve performance, we incorporate lexical matching.

### Idea

Combine:

* Semantic similarity (SBERT)
* Lexical similarity (e.g., TF-IDF, BM25)

### Final Score

```
final_score = α * semantic_score + β * lexical_score
```

### Benefits

* Captures exact word overlap
* Improves ranking accuracy

---

## Project Structure

```
project/
│
├── data/
│   ├── collection.tsv
│   ├── queries.train.tsv
│   ├── queries.dev.tsv
│   ├── qrels.train.tsv
│   └── qrels.dev.tsv
│
├── preprocess.py
├── data_loader.py
├── train.py
├── evaluate.py
│
├── train_pairs.json
├── dev_candidates.json
│
└── README.md
```

---

## Key Design Decisions

### 1. Negative Sampling

* Prevents model from learning trivial patterns
* Improves decision making between right and wrong examples

### 2. Query-Based Grouping

* Ensures model learns relevance per query

### 3. Subsampling

* Speeds up training and provides a richer representation by randomly removing high frequency words

---