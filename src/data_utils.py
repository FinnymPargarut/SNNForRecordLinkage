import pandas as pd
import yaml
from faker import Faker
import torch
from torch.utils.data import Dataset, random_split
from data_augmentation import augmentation


def char_to_index(char):
    """Converts characters to index its character embeddings"""
    if 'a' <= char <= 'z':
        return ord(char) - ord('a') + 1
    elif 'а' <= char <= 'я':
        return ord(char) - ord('а') + 27
    elif char.isdigit():
        return ord(char) - ord('0') + 60
    elif char == ' ':
        return 70
    elif char == ',':
        return 71
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


def create_record_list(records_count=100):
    """Create record list (csv file) in data directory"""
    fake = Faker('ru_RU')
    d = {
        'Name': [fake.name() for _ in range(records_count)],
        'Email': [fake.email() for _ in range(records_count)],
        'Phone': [fake.phone_number() for _ in range(records_count)]
    }
    data = pd.DataFrame(d)
    data.to_csv("../data/record-list.csv", index=False)


def post_process_data(str):
    """Lower case all characters of string"""
    return ''.join(c.lower() for c in str)


def get_pairs_labels_with_augmentation():
    """
    Get augmented pairs and labels from record list in data directory.
    The labels contains balanced classes, meaning that the number of zeros and ones is equal.

    The data in each pair is processed as follows:
    1. Augmentation is applied to the name, email, and phone.
    2. The augmentation results are concatenated into a single string.
    3. All characters in the string are converted to lowercase (as postprocessing).
    """
    data = pd.read_csv("../data/record-list.csv")
    pairs = []
    labels = []
    for i in range(len(data)):
        for j in range(i, len(data)):
            aug_count = 198 if i == j else 4  # It needs to balance classes
            for _ in range(aug_count):
                first_elem = ",".join(augmentation(data["Name"].iloc[i], data["Email"].iloc[i], data["Phone"].iloc[i]))
                second_elem = ",".join(augmentation(data["Name"].iloc[j], data["Email"].iloc[j], data["Phone"].iloc[j]))

                first_elem = post_process_data(first_elem)
                second_elem = post_process_data(second_elem)

                pairs.append((first_elem, second_elem))
                labels.append(1 if i == j else 0)

    return pairs, labels


def save_train_test_split(check_labels_distribution=True):
    """Function uses random_split from pytorch to create and save train and test dataset"""
    pairs, labels = get_pairs_labels_with_augmentation()
    with open("config.yml", "r") as options:
        max_len = yaml.safe_load(options)["input_data"]["max_len"]
    dataset = RecordLinkageDataset(pairs, labels, max_len)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    if check_labels_distribution:
        train_labels = torch.tensor([label for _, _, label in train_dataset])
        test_labels = torch.tensor([label for _, _, label in test_dataset])
        print("Train labels distribution:", torch.bincount(train_labels))
        print("Test labels distribution:", torch.bincount(test_labels))

    torch.save(train_dataset, "../data/train_dataset.pt")
    torch.save(test_dataset, "../data/test_dataset.pt")


if __name__ == '__main__':
    # create_record_list()
    save_train_test_split()
