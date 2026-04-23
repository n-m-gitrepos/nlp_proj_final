import os
import random
import json
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")

NUM_QUERIES = 2000        # sample NUM_QUERIES queries
NEGATIVES = 4            # no. of negative candidates per query
MAX_POSITIVES = 5        # no. relevant passages per query
DEV_CANDIDATES = 100     # candidates per dev query

random.seed(42)

# -----------------------------
# 1. LOAD DATA
# -----------------------------
def load_passages(path):
    '''
    Loads passages from collections.tsv.

    Format:
        passage_id \t passage_text

    Returns:
        dict: {passage_id: passage_text}
    '''
    passages = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            pid, text = line.strip().split('\t', 1)
            passages[pid] = text
    return passages

def load_queries(path):
    '''
    Loads queries from queries.tsv (dev, eval or train).

    Format:
        query_id \t query_text

    Returns:
        dict: {query_id: query_text}
    '''
    queries = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            qid, text = line.strip().split('\t', 1)
            queries[qid] = text
    return queries

def load_qrels(path):
    '''
    Loads ground truths from qrels.tsv (dev or train).

    Format:
        query_id \t 0 \t passage_id \t relevance

    Only relevance = 1 is considered (positive examples).

    Returns:
        dict: {query_id: set(relevant passages)}
    '''
    qrels = defaultdict(set)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            qid, _, pid, rel = line.strip().split('\t')
            if rel == '1':
                qrels[qid].add(pid)
    return qrels

# -----------------------------
# 2. NEGATIVE SAMPLING
# -----------------------------
def sample_negatives(passage_ids, positive_set, k):
    '''
    Randomly samples k negative passages. Use this helper function to conduct negative sampling.

    A negative passage is any passage NOT in the positive set.

    Args:
        passages_ids (list): all possible passage IDs
        positive_set (set): relevant passage IDs for a query
        k (int): number of negatives to sample

    Returns:
        list: sampled negative passage IDs
    '''
    negatives = []
    # sample until we have k negatives
    while len(negatives) < k:
        pid = random.choice(passage_ids)
        # only add if sample is not positive
        if pid not in positive_set:
            negatives.append(pid)
    return negatives

# -----------------------------
# 3. CREATE TRAINING PAIRS
# -----------------------------
def create_training_data(queries, passages, qrels):
    '''
    Creates training data for Sentence-BERT.

    Only queries with at least MAX_POSITIVES relevant passages are included.
    For each qualifying query, exactly MAX_POSITIVES positives are sampled,
    and NEGATIVES negatives are sampled per positive (25 passages total).
 
    Returns:
        dict: {
            query_id: {
                'query': str,
                'passages': [
                    {'passage': str, 'label': int (1=positive, 0=negative)},
                    ...  # 5 positives + 4 negatives each = 25 total
                ]
            }
        }
    '''
    passage_ids = list(passages.keys())

    # Only keep queries that have at least MAX_POSITIVES relevant passages
    eligible_qids = [qid for qid in qrels if len(qrels[qid]) >= MAX_POSITIVES]

    # Randomly sample NUM_QUERIES from eligible queries
    sampled_qids = random.sample(eligible_qids, min(NUM_QUERIES, len(eligible_qids)))

    training_data = {}

    for qid in sampled_qids:
        query = queries[qid]
        all_positives = qrels[qid]

        positives = set(random.sample(list(all_positives), MAX_POSITIVES))
        passages_list = []

        for pid in positives:
            # positive pair
            passages_list.append({
                'passage': passages[pid],
                'label': 1
            })

            # sample NEGATIVES negatives for each positive passage
            neg_pids = sample_negatives(passage_ids, all_positives, NEGATIVES)

            for neg_pid in neg_pids:
                passages_list.append({
                    'passage': passages[neg_pid],
                    'label': 0
                })
        training_data[qid] = {
            'query': query,
            'passages': passages_list
        }
        
    return training_data

# -----------------------------
# 4. CREATE DEV EVALUATION SET
# -----------------------------
def create_dev_data(queries, passages, qrels):
    '''
    Creates the evaluation dataset.

     For each query:
        - Sample MAX_POSITIVES positives
        - Sample remaining negatives to reach DEV_CANDIDATES
        - Shuffle candidates

    Returns:
        dict:
        {
            query_id: {
                'query': str,
                'candidates': [
                    {'passage': str, 'label': int (0/1)}
                ]
            }
        }
    '''
    passage_ids = list(passages.keys())
    dev_data = {}

    # Only keep queries with enough positives
    eligible_qids = [qid for qid in qrels if len(qrels[qid]) >= MAX_POSITIVES]

    for qid in eligible_qids:
        if qid not in queries:
            continue

        query = queries[qid]
        all_positives = list(qrels[qid])

        # --- sample fixed number of positives ---
        positives = random.sample(all_positives, MAX_POSITIVES)

        candidates = []

        # add positives
        for pid in positives:
            candidates.append({
                'passage': passages[pid],
                'label': 1
            })

        # --- sample unique negatives ---
        num_negatives = DEV_CANDIDATES - MAX_POSITIVES

        negative_pool = list(set(passage_ids) - set(all_positives))
        negatives = random.sample(negative_pool, num_negatives)

        for pid in negatives:
            candidates.append({
                'passage': passages[pid],
                'label': 0
            })

        # --- shuffle candidates (important!) ---
        random.shuffle(candidates)

        dev_data[qid] = {
            'query': query,
            'candidates': candidates
        }

    return dev_data

# -----------------------------
# MAIN
# -----------------------------

def main():
    '''
    Main preprocessing pipeline:
        1. Load all datasets
        2. Create training data
        3. Create evaluation (dev) data
        4. Save results to JSON files
    '''
    print("Loading data...")

    # Load passages from collection
    passages = load_passages(os.path.join(DATA_DIR, "collection.tsv"))

    # Load queries (train and dev)
    queries_train = load_queries(os.path.join(DATA_DIR, "queries.train.tsv"))
    queries_dev = load_queries(os.path.join(DATA_DIR, "queries.dev.tsv"))

    # Load qrels (train and dev)
    qrels_train = load_qrels(os.path.join(DATA_DIR, "qrels.train.tsv"))
    qrels_dev = load_qrels(os.path.join(DATA_DIR, "qrels.dev.tsv"))

    print("Creating training data...")
    train_data = create_training_data(queries_train, passages, qrels_train)

    print("Creating dev data...")
    dev_data = create_dev_data(queries_dev, passages, qrels_dev)

    print("Saving outputs...")

    # Create processed_data directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save training pairs
    with open(os.path.join(OUTPUT_DIR, "train_pairs.json"), "w") as f:
        json.dump(train_data, f, indent=2)

    # Save evaluation candidates
    with open(os.path.join(OUTPUT_DIR, "dev_candidates.json"), "w") as f:
        json.dump(dev_data, f, indent=2)

    print("Done!")
    print(f"Training samples: {len(train_data)}")
    print(f"Dev queries: {len(dev_data)}")


if __name__ == "__main__":
    main()