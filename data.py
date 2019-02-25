import numpy as np
import string
import pickle
import os


def save_object(obj, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open("last_object_saved", "w") as f:
        f.write(path)


def load_object(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def flip_dict(map):
    return {v: k for k, v in map.items()}


def encodings(text, ix_to_char):
    if ix_to_char is None:
        ix_to_char = dict(enumerate(set(text)))
    char_to_ix = flip_dict(ix_to_char)
    return char_to_ix, ix_to_char


def read_corpus(data_path):
    with open(data_path, "r") as f:
        return f.read()


def process_text(data_path, lower=True, remove_punctuation=False):
    raw = read_corpus(data_path)
    out = raw
    if lower:
        out = out.lower()
    if remove_punctuation:
        out = strip_punctuation(out)
    return out


def prepare_numeric(processed_text, char_to_ix):
    return [char_to_ix[ch] for ch in processed_text]


def data_split(numeric, num_sites, fraction):
    num_phrases = len(numeric) // num_sites
    num_train_phrases = round(num_phrases * fraction)
    train_size = num_train_phrases * num_sites
    train = np.array(numeric[:train_size]).reshape(num_train_phrases, num_sites)
    cv = np.array(numeric[train_size:2*train_size]).reshape(num_train_phrases, num_sites)
    population = np.array(numeric[2*train_size:3*train_size]).reshape(num_train_phrases, num_sites)
    return train, cv, population


def strip_punctuation(phrase):
    translator = str.maketrans('', '', string.punctuation)
    return phrase.lower().translate(translator)