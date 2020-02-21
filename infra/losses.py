import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

    
def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.a
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    np_true_mask = np.array(true.detach().cpu())
    true_mask_int = torch.zeros(true.shape).type(torch.int64).to(true.device)
    true_mask_int[np_true_mask.nonzero()] =  1
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true_mask_int.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).double()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true_mask_int.squeeze(1)].to(true.device)
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).double()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())

    intersection = torch.sum(probas * true_1_hot)
    cardinality = torch.sum(probas + true_1_hot)
    dice_loss = (2. * intersection / (cardinality + eps))
    return (1 - dice_loss)    