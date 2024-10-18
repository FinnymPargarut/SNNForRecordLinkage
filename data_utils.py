from torch.utils.data import Dataset
from siamese_neural_network import *


def char_to_index(char):
    """Converts characters to index its character embeddings"""
    if 'a' <= char <= 'z':
        return ord(char) - ord('a') + 1
    elif 'A' <= char <= 'Z':
        return ord(char) - ord('A') + 27
    elif 'а' <= char <= 'я':
        return ord(char) - ord('а') + 53
    elif 'А' <= char <= 'Я':
        return ord(char) - ord('А') + 79
    elif char.isdigit():
        return ord(char) - ord('0') + 105
    elif char == ' ':
        return 115
    else:
        return 0


def record_to_tensor(record, max_len):
    """Converts a record into a tensor with index its character embeddings"""
    tensor = torch.zeros(max_len, dtype=torch.long)
    for i, char in enumerate(record[:max_len]):
        tensor[i] = char_to_index(char)
    return tensor


class RecordLinkageDataset(Dataset):
    """
    Class for representing a dataset for the record linkage task.

    This class inherits from `torch.utils.data.Dataset` and is designed to work with pairs of records and their labels.
    Each record pair represents a potential match or non-match.

    Args:
        pairs (list): A list of record pairs, where each pair is represented as a tuple (record1, record2).
        labels (list): A list of labels corresponding to the record pairs - 1 for a match and 0 for a non-match.
        max_len (int): The maximum length of a record, used to pad all records to the same size.

    Methods:
        __len__: Returns the number of record pairs in the dataset.
        __getitem__: Returns a pair of tensors representing the records and the corresponding label for a given index.
    """
    def __init__(self, pairs, labels, max_len):
        self.pairs = pairs
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        record1, record2 = self.pairs[idx]
        label = self.labels[idx]
        tensor1 = record_to_tensor(record1, self.max_len)
        tensor2 = record_to_tensor(record2, self.max_len)
        return tensor1, tensor2, label


def create_pairs(records):
    """Create pair records and their labels (placeholder)"""
    pairs = []
    labels = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            pairs.append((records[i], records[j]))
            labels.append(1 if i == j else 0)
    return pairs, labels
