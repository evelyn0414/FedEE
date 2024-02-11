from alg.fedavg import fedavg
import torch.nn as nn
import torch.optim as optim

class fedbn(fedavg):
    def __init__(self, args, model=None, loss=nn.CrossEntropyLoss(), optimizer=optim.SGD):
        super(fedbn, self).__init__(args, model, loss, optimizer)
