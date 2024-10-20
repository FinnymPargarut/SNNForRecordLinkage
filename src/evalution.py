import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from siamese_predict import get_SNN
from data_utils import RecordLinkageDataset  # used in torch.load


def counter():
    """
    Calculates and prints the average distances for labels 0 and 1 in the dataset.
    It needs to choose threshold for classification.
    """
    distances_label_0 = []
    distances_label_1 = []

    dataset = torch.load("../data/train_dataset.pt")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    c = 0

    SNN = get_SNN()
    for tensor1, tensor2, label in dataloader:
        c += 1
        with torch.no_grad():
            output1, output2 = SNN(tensor1, tensor2)
        distance = F.pairwise_distance(output1, output2).item()
        if label == 0:
            distances_label_0.append(distance)
        elif label == 1:
            distances_label_1.append(distance)
    avg_distance_label_0 = sum(distances_label_0) / len(distances_label_0) if distances_label_0 else 0
    avg_distance_label_1 = sum(distances_label_1) / len(distances_label_1) if distances_label_1 else 0
    print(f"The count of elements in the dataset: {c}")
    print(f"The average distance for the label 0: {avg_distance_label_0}")
    print(f"The average distance for the label 1: {avg_distance_label_1}")


def evaluate_model(threshold):
    """
    Evaluates the model's performance on the test dataset using the given threshold.
    Prints accuracy, precision, recall, and F1 score.
    """
    dataset = torch.load("../data/test_dataset.pt")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    SNN = get_SNN()

    true_labels = []
    predicted_labels = []

    for tensor1, tensor2, label in dataloader:
        with torch.no_grad():
            output1, output2 = SNN(tensor1, tensor2)
        distance = F.pairwise_distance(output1, output2).item()

        predicted_label = 1 if distance < threshold else 0

        true_labels.append(label.item())
        predicted_labels.append(predicted_label)

    true_labels = torch.tensor(true_labels)
    predicted_labels = torch.tensor(predicted_labels)

    accuracy = (predicted_labels == true_labels).float().mean().item()
    precision = (predicted_labels[true_labels == 1] == 1).float().mean().item()
    recall = (true_labels[predicted_labels == 1] == 1).float().mean().item()
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


if __name__ == '__main__':
    evaluate_model(0.0002)
