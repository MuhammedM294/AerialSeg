import torch

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Intersection over Union (IoU) metric for binary segmentation masks using PyTorch.

    Args:
        outputs (torch.Tensor): Predicted segmentation masks.
            Shape: (BATCH_SIZE, 1, H, W), where BATCH_SIZE is the number of samples, H is the height of the mask, and W is the width of the mask.
        labels (torch.Tensor): Ground truth segmentation masks.
            Shape: (BATCH_SIZE, 1, H, W), where BATCH_SIZE is the number of samples, H is the height of the mask, and W is the width of the mask.

    Returns:
        torch.Tensor: Mean IoU score for the batch of samples.

    The IoU (also known as Jaccard index) is a commonly used evaluation metric for image segmentation tasks.
    It measures the overlap between the predicted and ground truth masks, providing a measure of the segmentation accuracy.

    The function calculates the IoU score by first converting the outputs and labels tensors to 2D masks (BATCH_SIZE x H x W).
    Then, it computes the intersection and union of the masks, taking into account only the foreground pixels (where the value is 1).
    Finally, it applies smoothing to the division to avoid division by zero and returns the mean IoU score for the batch of samples.

    Note:
        - The function assumes that both the outputs and labels tensors represent binary masks (with pixel values of 0 or 1).
        - The tensors should have the same shape and batch size.


    """
    SMOOTH = 1e-6
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our division to avoid 0/0

    return iou.mean()


def pixel_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes the accuracy for multiple binary predictions.

    Args:
        output (torch.Tensor): Predicted values or logits.
            Shape: (BATCH_SIZE, ...)
        target (torch.Tensor): Ground truth labels or targets.
            Shape: (BATCH_SIZE, ...)

    Returns:
        float: Accuracy score.

    The function computes the accuracy by comparing the predicted values or logits with the ground truth labels or targets.
    It rounds the predictions to 0 or 1 by applying the sigmoid function, and then compares them with the targets.
    The number of correct predictions is counted, and the accuracy is calculated as the ratio of correct predictions to the total number of targets.

    Note:
        - The function assumes that both the output and target tensors represent binary predictions or labels.
        - The tensors can have any shape, as long as they have the same dimensions.


    """
    with torch.no_grad():
        # Round predictions to 0 or 1
        pred = torch.round(torch.sigmoid(output))
        correct = pred.eq(target).sum().float()
        accuracy = correct / len(target)
        return accuracy

