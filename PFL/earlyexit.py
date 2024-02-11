import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ensemble_weights = torch.tensor([1, 1, 1, 1])


class ExitBlock3layer(nn.Module):

    def __init__(self, in_channels, pool_dim, hidden_sizes, out_channels):
        super().__init__()
        
        layers = [nn.AdaptiveAvgPool2d(pool_dim)]
        layers += [nn.Flatten()]
        layers += [nn.Linear(in_channels*pool_dim*pool_dim, hidden_sizes)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(hidden_sizes, int(hidden_sizes * 0.5))]
        layers += [nn.ReLU()]
        layers += [nn.Linear(int(hidden_sizes * 0.5), out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        return self.layers(x)


class ExitBlockAdaptive(nn.Module):

    def __init__(self, in_channels, pool_dim, hidden_sizes, out_channels):
        super().__init__()
        
        layers = [nn.AdaptiveAvgPool2d(pool_dim)]
        layers += [nn.Flatten()]
        layers += [nn.Linear(in_channels*pool_dim*pool_dim, hidden_sizes)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(hidden_sizes, out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        return self.layers(x)


class ExitBlock2D(nn.Module):

    def __init__(self, in_channels, hidden_sizes, out_channels, binary=False):
        super().__init__()

        layers = [nn.AdaptiveAvgPool2d(1)]
        layers += [nn.Flatten()]
        layers += [nn.Linear(in_channels, hidden_sizes)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(hidden_sizes, out_channels)]
        # if binary:
        #     layers += [nn.Sigmoid()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        return self.layers(x)


class ExitBlock(nn.Module):

    def __init__(self, in_channels, hidden_sizes, out_channels):
        super().__init__()

        layers = [nn.AdaptiveAvgPool1d(1)]
        layers += [nn.Flatten(1)]
        layers += [nn.Linear(in_channels, hidden_sizes)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(hidden_sizes, out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        return self.layers(x)
       

class ExitBlockP(nn.Module):

    def __init__(self, in_channels, hidden_sizes):
        super().__init__()

        layers = [nn.AdaptiveAvgPool1d(1)]
        layers += [nn.Flatten(1)]
        layers += [nn.Linear(in_channels, hidden_sizes)]
        layers += [nn.ReLU()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        return self.layers(x)


def F1(logits, labels, ensemble_weights=torch.tensor([1, 1, 1, 1]), average="weighted"):

    _, num_exits, num_classes = logits.shape
    scale = ensemble_weights.sum()

    pred_labels = logits.softmax(dim=-1).mul(ensemble_weights).sum(dim=-2).div(scale).argmax(-1)

    f1 = tm.f1(pred_labels, labels, num_classes=num_classes, average=average)

    return f1


def negative_loglikelihood(logits, labels, ensemble_weights=torch.tensor([1, 1, 1, 1]), reduction="mean"):

    _, num_exits, num_classes = logits.shape
    scale = ensemble_weights.sum()

    probs = logits.softmax(dim=-1).mul(ensemble_weights).sum(dim=-2).div(scale)

    nll = -Categorical(probs=probs).log_prob(labels)

    if reduction == "mean":
        nll = nll.mean()

    return nll


def brier_score(logits, labels, ensemble_weights=torch.tensor([1, 1, 1, 1]), reduction="mean"):
    # _,  num_classes = logits.shape
    _, num_exits, num_classes = logits.shape
    scale = ensemble_weights.sum()

    probs = logits.softmax(dim=-1).mul(ensemble_weights).sum(dim=-2).div(scale)

    labels_one_hot = F.one_hot(labels, num_classes=num_classes)

    bs = ((probs - labels_one_hot)**2).sum(dim=-1)

    if reduction == "mean":
        bs = bs.mean()

    return bs
    

def expected_calibration_error(logits, labels, ensemble_weights=torch.tensor([1, 1, 1, 1]), n_bins=15):

    num_samples, num_exits, num_classes = logits.shape
    scale = ensemble_weights.sum() 
    # pred_probs = logits.softmax(dim=-1)
    pred_probs = logits.softmax(dim=-1).mul(ensemble_weights).sum(dim=-2).div(scale)
    pred_labels = pred_probs.argmax(-1)

    pred_probs = pred_probs[torch.arange(num_samples), pred_labels]

    correct = pred_labels.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    conf_bin = torch.zeros_like(bin_boundaries)
    acc_bin = torch.zeros_like(bin_boundaries)
    prop_bin = torch.zeros_like(bin_boundaries)
    
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):

        in_bin = pred_probs.gt(bin_lower.item()) * pred_probs.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            # probability of making a correct prediction given a probability bin
            acc_bin[i] = correct[in_bin].float().mean()
            # average predicted probabily given a probability bin.
            conf_bin[i] = pred_probs[in_bin].mean()
            # probability of observing a probability bin
            prop_bin[i] = prop_in_bin
    
    ece = ((acc_bin - conf_bin).abs() * prop_bin).sum()

    return ece


def calculate_metrics(model, logits, labels, ensemble_weights=torch.tensor([1, 1, 1, 1])):

    metrics = dict(f1=F1(logits, labels, ensemble_weights, average="weighted").numpy(),
                #   precision=precision(logits, labels, ensemble_weights, average="weighted").numpy(),
                #   recall=recall(logits, labels, ensemble_weights, average="weighted").numpy(),
                   negative_loglikelihood=negative_loglikelihood(logits, labels, ensemble_weights, reduction="mean").numpy(),
                   brier_score=brier_score(logits, labels, ensemble_weights, reduction="mean").numpy(),
                #   predictive_entropy=predictive_entropy(logits, labels, ensemble_weights, reduction="mean").numpy(),
                   expected_calibration_error=expected_calibration_error(logits, labels, ensemble_weights, n_bins=15).numpy())

    return metrics