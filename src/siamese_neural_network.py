import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
            The ContrastiveLoss class implements a contrastive loss function, which is used for training networks like
            Siamese Network. The contrastive loss function is designed to minimize the distance between instances
            of the same class and maximize the distance between instances of different classes.

            Args:
                margin (float): A parameter that defines the minimum distance between instances of different classes.
                                Defaults to 1.0.
        """
        super(ContrastiveLoss, self).__init__()

        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: float) -> torch.Tensor:
        """
        Computes the contrastive loss for two output tensors and a label.

        Args:
            output1 (torch.Tensor): Output from the first subnetwork.
            output2 (torch.Tensor): Output from the second subnetwork.
            label (float): A label indicating whether the instances are of the same class (0) or different (1).

        Returns:
            torch.Tensor: The contrastive loss value.
        """
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class TransformerSubnetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        """
            The TransformerSubnetwork class implements a Transformer-based subnetwork, which can be used in a Siamese Network.

            Args:
                input_dim (int): Dimension of the input vocabulary (number of unique tokens).
                hidden_dim (int): Dimension of the hidden layer (d_model in Transformer).
                num_layers (int): Number of layers in the TransformerEncoder.
                num_heads (int): Number of heads in the attention mechanism.
                dropout (float): Dropout probability for regularization.
        """
        super(TransformerSubnetwork, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
            Performs a forward pass through the Transformer subnetwork.

            Args:
                x (torch.Tensor): Input tensor with size (batch_size, sequence_length).

            Returns:
                torch.Tensor: Output tensor with size (batch_size, hidden_dim).
        """
        x = self.embedding(x)
        x = self.transformer(x)
        x = torch.mean(x, dim=1) # Mean over the sentence
        x = self.fc(x)
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        """
            The SiameseNetwork class implements a Siamese Network, which consists of two identical Transformer-based
            subnetworks. The network is used for tasks where it is necessary to compare
            two instances and determine how similar they are.

            Args:
                input_dim (int): Dimension of the input vocabulary (number of unique tokens).
                hidden_dim (int): Dimension of the hidden layer (d_model in Transformer).
                num_layers (int): Number of layers in the TransformerEncoder.
                num_heads (int): Number of heads in the attention mechanism.
                dropout (float): Dropout probability for regularization.
        """
        super(SiameseNetwork, self).__init__()
        self.subnetwork = TransformerSubnetwork(input_dim, hidden_dim, num_layers, num_heads, dropout)

    def forward(self, input1, input2):
        """
            Performs a forward pass through the Siamese Network for two input tensors.

            Args:
                input1 (torch.Tensor): First input tensor with size (batch_size, sequence_length).
                input2 (torch.Tensor): Second input tensor with size (batch_size, sequence_length).

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: A tuple of two output tensors, each with size (batch_size, hidden_dim).
        """
        output1 = self.subnetwork(input1)
        output2 = self.subnetwork(input2)
        return output1, output2
