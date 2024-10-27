import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from importlib.metadata import version

from task import *

partition_id = 0
num_partitions = 10
batch_size = 32
device = "cpu"
epochs = 10
learning_rate = 0.1

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

class TextDataset(Dataset):
    def __init__(self, data, vocab, tokenizer):
        self.data = data
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        tokens = self.tokenizer(text)
        token_ids = [self.vocab[token] for token in tokens]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_batch(batch):
    text_list, label_list = [], []
    for (_text, _label) in batch:
        text_list.append(_text)
        label_list.append(_label)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab['<pad>'])
    label_list = torch.tensor(label_list, dtype=torch.long)
    return text_list, label_list

def train_LSTM(device: str = "cpu"):
    # Prepare data
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="sentiment140",
        partitioners={"train": partitioner},
    )
    print('fds: ', fds)

    # Load partition
    partition = fds.load_partition(partition_id)
    print('partition: ', partition)

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    print('partition_train_test: ', partition_train_test)

    # Example data
    train_data = [(data['text'], data['sentiment']) for data in partition_train_test['train']]
    test_data = [(data['text'], data['sentiment']) for data in partition_train_test['test']]

    # Tokenizer and vocabulary
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(map(tokenizer, [text for text, _ in train_data]), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])

    # Datasets and DataLoaders
    train_dataset = TextDataset(train_data, vocab, tokenizer)
    test_dataset = TextDataset(test_data, vocab, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch)

    # Train
    net = StackedLSTM()
    net.to(device)
    print("src/fedml training on", device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for text, labels in train_loader:
            optimizer.zero_grad()
            print('text: ', text)
            print('labels: ', labels)
            criterion(net(text.to(device)), labels.to(device)).backward()
            optimizer.step()
        val_loss, val_acc = test(net, test_loader, device)
        print(f"Epoch {_}, val_loss: {val_loss}, val_acc: {val_acc}")

if __name__ == "__main__":
    version('torchtext')
    train_LSTM()