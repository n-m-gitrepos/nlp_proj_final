from sentence_transformers import InputExample
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "processed_data", "train_pairs.json")

def load_training_examples(path):
    '''
    Loads training examples from train_pairs.json and converts it
    into Sentence-BERT InputExample format.
    
    Args:
        path (str): path to train_pairs.json

    Returns:
        list[InputExample]: list of training examples
    '''
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    examples = []

    for item in data:
        query = item['query']
        passage = item['passage']
        label = item['label']

        examples.append(
            InputExample(
                texts=[query, passage],
                label=label
            )
        )
    return examples

if __name__ == "__main__":
    train_examples = load_training_examples(DATA_PATH)

    print(f"Loaded {len(train_examples)} training examples")

    # sanity check
    print(train_examples[0])