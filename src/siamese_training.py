import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from siamese_neural_network import *
from data_utils import RecordLinkageDataset  # used in torch.load


class SiameseTraining:
    def __init__(self, model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                 optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
                 device: torch.device):
        """
        Initializes the training procedure for the models network.

        Args:
            model (nn.Module): The models network model.
            dataloader (DataLoader): DataLoader for training data.
            criterion (nn.Module): Loss function to use during training.
            optimizer (torch.optim.Optimizer): Optimizer to update model parameters.
            scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler.
            device (torch.device): Device to use for training (e.g., cuda or cpu).
        """
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, current_epoch: int, epochs: int, enable_prints: bool = True, print_every: int = 10):
        """
        Train the models network for one epoch.

        Args:
            current_epoch (int): The current epoch number.
            epochs (int): Total number of epochs to train the model.
            enable_prints (bool): If True, print loss during training.
            print_every (int): Print the loss every 'print_every' steps.
        """
        total_loss = 0.0
        for i, data in enumerate(self.dataloader):
            input1, input2, label = data

            input1 = input1.to(self.device)
            input2 = input2.to(self.device)
            label = label.to(self.device)

            self.optimizer.zero_grad()
            output1, output2 = self.model(input1, input2)
            loss = self.criterion(output1, output2, label.float())
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if enable_prints and i % print_every == 0:
                print(f'Epoch [{current_epoch + 1}/{epochs}], Item [{i}/{len(self.dataloader)}], '
                      f'Loss: {loss.item():.6f}')

        avg_loss = total_loss / len(self.dataloader)

        self.scheduler.step(avg_loss)

        print(f'\033[32mEpoch [{current_epoch + 1}/{epochs}], Avg Loss: {avg_loss:.6f}\033[0m')


if __name__ == '__main__':
    with open("config.yml", "r") as options:
        args = yaml.safe_load(options)["SNN_args"]

    dataset = torch.load("../data/train_dataset.pt")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initializing model and training components
    model = SiameseNetwork(**args)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Starting training
    snn_training = SiameseTraining(model, dataloader, criterion, optimizer, scheduler, device)
    model.train()
    num_epochs = 40
    save_every = 10
    for epoch in range(num_epochs):
        snn_training.train(epoch, num_epochs, enable_prints=False)

        if epoch % save_every == 0:
            torch.save(model.state_dict(), f"../models/siamese_model_epoch_{epoch}.pth")
            torch.save(optimizer.state_dict(), f"../models/siamese_optimizer_epoch_{epoch}.pth")
            print(f"Saved model and optimizer state for epoch {epoch}")

    torch.save(model.state_dict(), "../models/siamese_model_final.pth")
    torch.save(optimizer.state_dict(), "../models/siamese_optimizer_final.pth")

    print("Training complete.")
