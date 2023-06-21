import torch.optim as optim
from tqdm import tqdm


def train_fn(train_dataloader, model, optimizer):
    """
    Trains the model using the provided training dataloader, model and optimizer.

    Args:
        train_dataloader (torch.utils.data.DataLoader): Dataloader for training data.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model's parameters.

    Returns:
        float: The average training loss per batch.


    """
    model.train()
    train_loss = 0
    for image, mask in tqdm(train_dataloader):
        optimizer.zero_grad()
        logits, loss = model(image, mask)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(train_dataloader)