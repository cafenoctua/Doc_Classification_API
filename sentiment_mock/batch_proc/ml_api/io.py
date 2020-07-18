import os
import csv
from random import choice
import string
from typing import List
from ml_api import schemas

storage = os.path.join(os.path.dirname(__file__), 'local_storage')

def save_csv(data, filepath: str, fieldnames=None):
    with open(filepath, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for f in data:
            writer.writerow(f)

def load_csv(filepath: str):
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        out = list(reader)
    return out

def save_inputs(data: schemas.Data, length=8):
    letters = string.ascii_lowercase
    filename = ''.join(choice(letters) for i in range(length)) + '.csv'
    filepath = os.path.join(storage, 'inputs', filename)
    save_csv(data=data.dict()['data'], filepath=filepath, fieldnames=['text'])
    return filename

def load_inputs(filename: str):
    filepath = os.path.join(storage, 'inputs', filename)
    texts = load_csv(filepath=filepath)
    texts = [schemas.Text(**f) for f in texts]
    return texts

def save_outputs(preds: List[str], filename):
    filepath = os.path.join(storage, 'outputs', filename)
    save_csv(data=preds, filepath=filepath, fieldnames=['text', 'sentiment'])
    return filename

def load_outputs(filename: str):
    filepath = os.path.join(storage, 'outputs', filename)
    return load_csv(filepath=filepath)

def check_outputs(filename: str):
    filepath = os.path.join(storage, 'outputs', filename)
    return os.path.exists(filepath)