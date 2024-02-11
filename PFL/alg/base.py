from alg.fedavg import fedavg
import torch.nn as nn
import torch.optim as optim

class base(fedavg):
    def __init__(self, args, model=None, loss=nn.CrossEntropyLoss(), optimizer=optim.SGD):
        super(base, self).__init__(args, model, loss, optimizer)

    def server_aggre(self):
        pass
