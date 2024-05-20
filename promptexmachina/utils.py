import json



def load_json(filepath, op='r'):

    with open(filepath, op) as f:
        content = json.load(f)

    return content


def save_json(filepath, content):

    with open(filepath, 'w') as f:
        json.dump(content, f)