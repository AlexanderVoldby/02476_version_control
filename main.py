import click
import torch
import matplotlib.pyplot as plt
from model import MyAwesomeModel

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--e", default=5, help="No. training epochs")
def train(lr, e):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()

    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(e):
        print(f"Epoch {i+1} of {e}")
        for images, labels in train_set:
            optim.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            loss.backward()
            optim.step()
    


    torch.save(model.state_dict(), 'checkpoint.pth')


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    _, test_set = mnist()
    with torch.no_grad():
        running_acc = 0
        for images, labels in test_set:
            out = model(images)
            top_p, top_c = out.topk(1, dim=1)
            equals = top_c == labels.view(*top_c.shape)
            running_acc += torch.mean(equals.type(torch.FloatTensor))

    accuracy = running_acc.item() / len(test_set)
    print(accuracy)
    return accuracy


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
