
import torch
from tqdm import tqdm
from src.metrics import iou_pytorch, pixel_accuracy


def eval_fn(valid_dataloader, model):
    """
    Evaluates the model using the provided validation dataloader and model. 

    Args:
        valid_dataloader (torch.utils.data.DataLoader): Dataloader for validation data.
        model (torch.nn.Module): The model to be evaluated.


    Returns:
        float: The average validation loss per batch.
        tensor: The average IoU per batch.
        float: The average pixel accuracy per batch.


    """
    model.eval()
    valid_loss = 0
    iou_sum = 0
    valid_dataloader_len = len(valid_dataloader)
    for image, mask in tqdm(valid_dataloader):
        with torch.no_grad():
            logits, loss = model(image, mask)
            valid_loss += loss.item()
            pred_mask = (torch.sigmoid(logits) > 0.5)*1
            iou = iou_pytorch(pred_mask, mask)
            iou_sum += iou
            accuracy = pixel_accuracy(pred_mask, mask)


    return valid_loss / valid_dataloader_len , iou_sum / valid_dataloader_len , accuracy