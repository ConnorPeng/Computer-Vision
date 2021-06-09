import torch
import torch.nn as nn
from torchvision.models import alexnet


class MyAlexNet(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Note: Do not forget to freeze the layers of alexnet except the last one. Otherwise the training will take a long time. To freeze a layer, set the
        weights and biases of a layer to not require gradients.

        Note: Map elements of alexnet to self.conv_layers and self.fc_layers.

        Note: Remove the last linear layer in Alexnet and add your own layer to 
        perform 15 class classification.

        Note: Download pretrained alexnet using pytorch's API (Hint: see the import statements)
        """
        super().__init__()

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

        ############################################################################
        # Student code begin
        ############################################################################

        raise NotImplementedError("AlexNet not implemented")

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Perform the forward pass with the net

        Note: do not perform soft-max or convert to probabilities in this function

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        model_output = None
        x = x.repeat(1, 3, 1, 1)  # as AlexNet accepts color images
        ############################################################################
        # Student code begin
        ############################################################################
        raise NotImplementedError("forward function for AlexNet not implemented")

        ############################################################################
        # Student code end
        ############################################################################

        return model_output
