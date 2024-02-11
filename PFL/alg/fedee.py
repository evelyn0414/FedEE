from alg.fedavg import fedavg
import torch.nn as nn
import torch.optim as optim

class fedee(fedavg):
    def __init__(self, args, model=None, loss=nn.CrossEntropyLoss(), optimizer=optim.SGD):
        super(fedee, self).__init__(args, model, loss, optimizer)
