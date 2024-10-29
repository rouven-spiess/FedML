"""pytorchexample: A Flower / PyTorch app."""

import typing

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fedml.task import Net, get_weights


# Define metric aggregation function
def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [
        num_examples * typing.cast(float, m["accuracy"]) for num_examples, m in metrics
    ]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""

    # Read from config
    num_rounds = int(context.run_config["num-server-rounds"])

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    fraction_evaluate = float(context.run_config["fraction-evaluate"])

    # Define the strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
