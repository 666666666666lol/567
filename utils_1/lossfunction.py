from torch import nn
import torch
import numpy


class FocalLoss(nn.Module):
    """
    FocalLoss is a loss function that addresses class imbalance by focusing more on hard-to-classify examples.
    It adjusts the standard binary cross entropy loss by adding a modulating factor (1 - p) ^ gamma.
    :param gamma: A parameter that reduces the loss for easily classified examples, focusing more on difficult examples.
    :param alpha: A coefficient for balancing the importance of positive and negative examples.
    :param r: A small value added to the logarithms for numerical stability.
    """

    def __init__(self, gamma=2, alpha=0.25, r=1e-19):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()
        self.r = r

    def forward(self, p, target):
        target = target.float()
        p_min = p.min()
        p_max = p.max()
        if p_min < 0 or p_max > 1:
            raise ValueError("Prediction values should be between [0, 1]")
        p = p.reshape(-1, 1)
        target = target.reshape(-1, 1)
        loss = -self.alpha * (1 - p) ** self.gamma * (
            target * torch.log(p + self.r)
        ) - (1 - self.alpha) * p**self.gamma * (
            (1 - target) * torch.log(1 - p + self.r)
        )
        return loss.mean()


class FocalLossManyClassification(nn.Module):
    """
    FocalLoss for multi-class classification, designed for softmax outputs.
    No need for additional softmax before using it.
    :param num_class: The number of classes.
    :param alpha: A list of coefficients to balance each class, with the length equal to the number of classes.
    :param gamma: A parameter to control the difficulty level for learning from hard examples.
    :param smooth: Label smoothing factor.
    :param epsilon: A small value added for numerical stability.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None):
        super(FocalLossManyClassification, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, numpy.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
            self.alpha = self.alpha.numpy()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError("Unsupported alpha parameter format")

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("Smooth factor should be between [0, 1]")

    def forward(self, input, target):
        logit = torch.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        epsilon = 1e-10
        alpha = self.alpha
        alpha = torch.tensor(alpha)
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        return loss.mean()


if __name__ == "__main__":
    f = FocalLossManyClassification(
        10, alpha=[1, 2, 15, 4, 8, 6, 7, 7, 9, 4], smooth=0.1
    )
    predict = torch.randn(64, 10, requires_grad=True)
    targets = torch.randint(0, 9, (64,))
    loss = f(torch.sigmoid(predict), targets)
    print(loss)
    loss.backward()
    # print(targets)
