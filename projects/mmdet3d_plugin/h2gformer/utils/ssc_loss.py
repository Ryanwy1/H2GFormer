import torch
import torch.nn as nn
import torch.nn.functional as F

def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term


def geo_scal_loss(pred, ssc_target):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )

def precision_loss(pred, ssc_target):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
    )

def sem_scal_loss(pred, ssc_target):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                

                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count

def CE_ssc_loss(pred, target, class_weights):

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="none"
    )
    loss = criterion(pred, target.long())
    loss_valid = loss[target!=255]
    loss_valid_mean = torch.mean(loss_valid)
    return loss_valid_mean

def BCE_ssc_loss(pred, target, class_weights, alpha):

    class_weights[0] = 1-alpha    # empty                 
    class_weights[1] = alpha    # occupied                      

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="none"
    )
    loss = criterion(pred, target.long())
    loss_valid = loss[target!=255]
    loss_valid_mean = torch.mean(loss_valid)

    return loss_valid_mean

def IoE_PA_loss(pred, target):
    cal_IoE = IoE().to(target.device)
    weights_IoE = cal_IoE(target)

    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="none")
    loss = criterion(pred, target.long())
    loss = weights_IoE * loss
    loss_valid = loss[target!=255]
    loss_valid_mean = torch.mean(loss_valid)
    return loss_valid_mean*5

class IoE(nn.Module):
    def __init__(self):
        super(IoE, self).__init__()
        kernel = [-1, 0, 1]
        # 1*1*3
        kernel1 = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # 1*3*1
        kernel2 = torch.FloatTensor(kernel).unsqueeze(1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # 3*1*1
        kernel3 = torch.FloatTensor(kernel).unsqueeze(1).unsqueeze(1).unsqueeze(0).unsqueeze(0)

        self.conv_1x1xk = nn.Conv3d(1, 1, (1, 1, 3), padding=(0, 0, 1), bias=False)
        self.conv_1xkx1 = nn.Conv3d(1, 1, (1, 3, 1), padding=(0, 1, 0), bias=False)
        self.conv_kx1x1 = nn.Conv3d(1, 1, (3, 1, 1), padding=(1, 0, 0), bias=False)

        self.conv_1x1xk.weight = torch.nn.Parameter(kernel1, requires_grad=False)
        self.conv_1xkx1.weight = torch.nn.Parameter(kernel2, requires_grad=False)
        self.conv_kx1x1.weight = torch.nn.Parameter(kernel3, requires_grad=False)

    def forward(self, x):
        x = x.unsqueeze(0)

        y1 = self.conv_1x1xk(x)
        y1[y1 != 0] = 1

        y2 = self.conv_1xkx1(x)
        y2[y2 != 0] = 1

        y3 = self.conv_kx1x1(x)
        y3[y3 != 0] = 1

        y = y1 + y2 + y3
        return y.squeeze(0)