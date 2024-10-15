import pytest
import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torch import Size, nn, optim


from src.fedml.task import Net, train, testing, load_data

@pytest.mark.skip
def test_load_CIFAR10():
    # Arrange
    num_partitions = 2
    partitioner = IidPartitioner(num_partitions=num_partitions)
    # Act
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )
    # Assert
    assert fds.load_partition(0).features['img'] is not None
    return fds

@pytest.mark.skip
def test_load_Sent140():
    # Arrange
    num_partitions = 2
    partitioner = IidPartitioner(num_partitions=num_partitions)
    # Act
    fds = FederatedDataset(
        dataset="Jean-Baptiste/financial_news_sentiment",
        partitioners={"train": partitioner},
    )
    # Assert
    assert fds.load_partition(0).features['topic'] is not None

# @pytest.mark.skip
def test_load_data_CIFAR10():
    # Act
    trainloader, testloader = load_data(0, 2, 32)
    # Assert
    assert trainloader.dataset['img'][0].shape == Size([3, 32, 32])
    assert testloader.dataset['img'][0].shape == Size([3, 32, 32])


def test_Net():
    # Arrange
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Act
    trainloader, valloader = load_data(partition_id=0, num_partitions=2, batch_size=32)
    # device = torch.tensor(trainloader)
    train(Net, trainloader, valloader, epochs=1, learning_rate=0.001, device=device)
    # test(net, valloader, device="cpu")





# def test_transform_Sent140():
#     assert True