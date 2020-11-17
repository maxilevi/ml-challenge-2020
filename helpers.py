import pickle
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

INTERACTIONS_PATH = './data/interactions.csv'
INTERACTIONS_TEST_PATH = './data/interactions_test.csv'
ITEMS_PATH = './data/items.csv'
ITEMS_PICKLE = './data/items.pickle'
UNCOMPRESSED_EMBEDDINGS_PATH = './data/embeddings/word2vec-uncompressed'
EMBEDDINGS_PATH = './data/embeddings/word2vec.bin'

def vectorize_items(items_dict):
    documents = [
        items_dict[x]['title'] for x in items_dict.keys()
    ]
    vectorizer = TfidfVectorizer(stop_words=['spanish', 'portuguese'], strip_accents='unicode')
    X = vectorizer.fit_transform(documents)
    return vectorizer, X

def load_interactions_df():
    return pd.read_csv(INTERACTIONS_PATH)

def load_interactions_test_df():
    return pd.read_csv(INTERACTIONS_TEST_PATH)

def load_items_df():
    return pd.read_csv(ITEMS_PATH)

def load_items():
    with open(ITEMS_PICKLE, 'rb') as handle:
        return pickle.load(handle)

def load_domain_item_dict(items_dict):
    domain_item_dict = {}
    for item in items_dict.keys():
        domain = items_dict[item]['domain_id']
        if domain not in domain_item_dict:
            domain_item_dict[domain] = []
        domain_item_dict[domain].append(item)
    return domain_item_dict

def load_top_items(interactions, domain_item_dict):
    counts = interactions['target'].value_counts()
    domain_top_items = {}
    for domain in domain_item_dict.keys():
        sorted_items = sorted([(x, counts.at[int(x)] if int(x) in counts.index else 0) for x in domain_item_dict[domain]], key=lambda x: -x[1])
        domain_top_items[domain] = [x[0] for x in sorted_items[:50]]
    return domain_top_items


def bin_array(n, m):
    return np.array(list(np.binary_repr(n).zfill(m))).astype(np.int8)

def pickle_items():
    item_dict = {}
    item_df = pd.read_json('./data/item_data.jl', lines=True, chunksize=500)
    def apply_row(line):
        item_dict[line.item_id] = {
            "title": line.title,
            "domain_id": line.domain_id,
            "product_id": line.product_id,
            "price": line.price,
            "category_id": line.category_id,
            "condition": line.condition,
        }

    for df in item_df:
        df.apply(apply_row, axis=1)

    print(f"Loaded {len(item_dict)} items into memory")

    with open(ITEMS_PICKLE, 'wb') as handle:
        pickle.dump(item_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _relevance(items_dict, item, target):
    if item == target:
        return 12
    elif items_dict[item]['domain_id'] == items_dict[target]['domain_id']:
        return 1
    return 0

def _ndcg(items_dict, recommendations, target):
    def get_perfect():
        perfect = [12, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        return sum(perfect[i] / math.log(i + 2) for i in range(len(perfect)))
    
    dcg = sum(_relevance(items_dict, recommendations[i], target) / math.log(i + 2) for i in range(len(recommendations)))
    return dcg / get_perfect()

def ndcg_score(recommendations, user_targets_dict):
    sum_ndcg = 0
    for x in recommendations.keys():
        sum_ndcg += _ndcg([int(w) for w in recommendations[x]], int(user_targets_dict[x]))

    return sum_ndcg / len(recommendations.keys())