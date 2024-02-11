# coding=utf-8
from alg.fedavg import fedavg
from util.traineval import train, train_prox
import torch.nn as nn
import torch.optim as optim


class fedprox(fedavg):
    def __init__(self, args, model=None, loss=nn.CrossEntropyLoss(), optimizer=optim.SGD):
        super(fedprox, self).__init__(args, model, loss, optimizer)

    def client_train(self, c_idx, dataloader, round):
        if round > 0:
            train_loss, train_acc = train_prox(
                self.args, self.client_model[c_idx], self.server_model, dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        else:
            train_loss, train_acc = train(
                self.client_model[c_idx], dataloader, self.optimizers[c_idx], self.loss_fun, self.args.device)
        return train_loss, train_acc
