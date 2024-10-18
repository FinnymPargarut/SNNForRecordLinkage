import yaml
from siamese_neural_network import *
from data_utils import record_to_tensor


def predict_duplicate(model, record1, record2, threshold=0.5):
    """Function to predict if two records are duplicates"""

    with open("config.yml", "r") as options:
        max_len = yaml.safe_load(options)["input_data"]["max_len"]

    tensor1 = record_to_tensor(record1, max_len)
    tensor2 = record_to_tensor(record2, max_len)

    tensor1 = tensor1.unsqueeze(0)
    tensor2 = tensor2.unsqueeze(0)

    with torch.no_grad():
        output1, output2 = model(tensor1, tensor2)

    euclidean_distance = F.pairwise_distance(output1, output2).item()
    print(euclidean_distance)

    if euclidean_distance < threshold:
        return True
    return False


def get_SNN():
    """Initialize and return Siamese Network"""
    with open("config.yml", "r") as options:
        args = yaml.safe_load(options)["SNN_args"]

    model = SiameseNetwork(**args)
    model.load_state_dict(torch.load("siamese/siamese_model_final.pth", weights_only=True))
    model.eval()

    return model


if __name__ == '__main__':
    SNN = get_SNN()
    r1 = "Jane Do"
    r2 = "John Do"
    is_duplicate = predict_duplicate(model=SNN, record1=r1, record2=r2)
    print(f"Записи являются дипликатами: {is_duplicate}")
