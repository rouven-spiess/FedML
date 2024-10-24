"""pytorchexample: A Flower / PyTorch app."""

import functools
import typing
from collections import OrderedDict
from collections.abc import Iterable

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.types
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class StackedLSTM(nn.Module):
    """StackedLSTM architecture.

    As described in Fei Chen 2018 paper :

    [FedMeta: Federated Meta-Learning with Fast Convergence and Efficient Communication]
    (https://arxiv.org/abs/1802.07876)
    """

    def __init__(self) -> None:
        super().__init__()

        self.embedding = nn.Embedding(80, 8)
        self.lstm = nn.LSTM(8, 256, num_layers=2, dropout=0.5, batch_first=True)
        self.fully_ = nn.Linear(256, 80)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """Forward pass of the StackedLSTM.

        Parameters
        ----------
        text : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        embedded = self.embedding(text)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embedded)
        return self.fully_(lstm_out[:, -1, :])


def get_weights(net: nn.Module) -> list[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters: Iterable[typing.Any]) -> None:
    params_dict = zip(net.state_dict(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class TrainTestDataLoaders(typing.NamedTuple):
    """A pair of train and test data loaders."""

    train: DataLoader
    test: DataLoader


def load_data(
    partition_id: int, num_partitions: int, batch_size: int
) -> TrainTestDataLoaders:
    partition_train_test = data_loader_CNN(partition_id, num_partitions)
    return data_transform_CNN(batch_size, partition_train_test)


@functools.lru_cache
def get_federated_dataset(num_partitions: int) -> FederatedDataset:
    partitioner = IidPartitioner(num_partitions=num_partitions)
    return FederatedDataset(
        dataset="uoft-cs/cifar10",
        # dataset="Jean-Baptiste/financial_news_sentiment",
        partitioners={"train": partitioner},
    )


def data_loader_CNN(partition_id: int, num_partitions: int) -> datasets.DatasetDict:
    """Load partition CIFAR10 data."""
    fds = get_federated_dataset(num_partitions)
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    print("partition_train_test", partition_train_test)
    return partition_train_test


def data_transform_CNN(
    batch_size: int,
    partition_train_test: datasets.DatasetDict,
) -> TrainTestDataLoaders:
    # TODO: adapt for LSTM
    pytorch_transforms = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return TrainTestDataLoaders(trainloader, testloader)


class TrainCNNResult(typing.TypedDict):
    val_loss: float
    val_accuracy: float


def train(
    net: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.types.Device,
) -> TrainCNNResult:
    return train_CNN(net, trainloader, valloader, epochs, learning_rate, device)


def train_CNN(
    net: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.types.Device,
) -> TrainCNNResult:
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()

    val_loss, val_acc = testing(net, valloader, device)

    return {
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }


def testing(
    net: nn.Module,
    testloader: DataLoader,
    device: torch.types.Device,
) -> tuple[float, float]:
    return testing_CNN(net, testloader, device)


def testing_CNN(
    net: nn.Module,
    testloader: DataLoader,
    device: torch.types.Device,
) -> tuple[float, float]:
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader)
    loss = loss / len(testloader)
    return loss, accuracy
