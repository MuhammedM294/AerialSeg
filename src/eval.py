
import torch
from tqdm import tqdm



def eval_fn(valid_dataloader, model, device):
    """
    Evaluates the model using the provided validation dataloader and model.

    Args:
        valid_dataloader (torch.utils.data.DataLoader): Dataloader for validation data.
        model (torch.nn.Module): The model to be evaluated.
        device (str): Device to be used for evaluation (e.g., 'cpu', 'cuda').

    Returns:
        float: The average validation loss per batch.

    """
    model.eval()
    valid_loss = 0
    for image, mask in tqdm(valid_dataloader):
        image = image.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            logits, loss = model(image, mask)
            valid_loss += loss.item()

    return valid_loss / len(valid_dataloader)