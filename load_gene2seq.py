import json

def load_gene2seq(path):
    with open(path) as f:
        gene2seq = json.load(f)
    return gene2seq
