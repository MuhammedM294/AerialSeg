import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import torch.nn as nn


class UNet(nn.Module):
    """
    A PyTorch implementation of the U-Net architecture for semantic segmentation.

    This class provides a wrapper around the U-Net model from the segmentation_models_pytorch library,
    allowing easy customization of the encoder, encoder weights, number of classes, and activation function.

    Args:
        encoder_name (str): The name of the pre-trained encoder to use. 
        encoder_weights (str): The source of the pre-trained encoder weights. 
        in_channels (int): The number of input channels. Defaults to 3.
        classes (int): The number of classes for the segmentation task. Defaults to 1.
        activation (str or callable): The activation function to use. Defaults to None.

    Attributes:
        model (nn.Module): The U-Net model instance from the segmentation_models_pytorch library.
    """
    def __init__(self, encoder_name:str, encoder_weights:str,in_channels:int = 3, classes:int = 1, activation:str = None):
        
        """
        Initializes a new instance of UNet.
        """
        
        super(UNet,self).__init__()

        self.model = smp.Unet( encoder_name=encoder_name,
                               encoder_weights=encoder_weights, 
                               in_channels=in_channels,
                               classes=classes,
                               activation=activation)
    
    def forward(self, images, masks = None):

        """
        Performs forward pass of the U-Net model.

        Args:
            images (torch.Tensor): Input images to be processed.
            masks (torch.Tensor, optional): Target masks for calculating the loss. Defaults to None.

        Returns:
            torch.Tensor: Logits produced by the U-Net model.
            torch.Tensor: Loss value if masks are provided, otherwise just the logits.
        """

        logits = self.model(images)
        if masks != None:
            loss = DiceLoss(mode='binary')(logits, masks)+ nn.BCEWithLogitsLoss()(logits, masks)
            return logits, loss
        return logits