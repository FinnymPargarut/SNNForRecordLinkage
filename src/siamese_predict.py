import yaml
from siamese_neural_network import *
from data_utils import record_to_tensor, post_process_data


def predict_duplicate(model, record1, record2, threshold=0.5):
    """Function to predict if two records are duplicates"""
    record1 = post_process_data(record1)
    record2 = post_process_data(record2)

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
    model.load_state_dict(torch.load("../models/siamese_model_final.pth", weights_only=True,
                                     map_location=torch.device('cpu')))
    model.eval()

    return model


if __name__ == '__main__':
    SNN = get_SNN()
    r1 = "Степанов Максимильян Трифонович,osipovjuvenali@example.org,8 (953) 492-67-32"
    r2 = "Носкова Евфросиния Матвеевна,ipatikostin@example.com,+7 878 256 83 73"
    is_duplicate = predict_duplicate(model=SNN, record1=r1, record2=r2, threshold=0.0002)
    print(f"Записи являются дубликатами: {is_duplicate}")
