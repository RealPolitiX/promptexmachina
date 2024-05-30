import json
import numpy as np
from collections import Counter


def load_json(filepath, op='r'):

    with open(filepath, op) as f:
        content = json.load(f)

    return content


def save_json(filepath, content):

    with open(filepath, 'w') as f:
        json.dump(content, f)


def calc_weights(arr):

    n = len(arr)
    grouping_counter = Counter(arr)
    # unique_groupings = list(grouping_counter.keys())
    grouping_weights = dict([(k, v/n) for k, v in grouping_counter.items()])
    weights = [grouping_weights[k] for k in arr]
    
    return weights