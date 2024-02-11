# from flamby.datasets.fed_tcga_brca import TcgaBrcaRaw, FedTcgaBrca
#
# # Raw dataset
# mydataset_raw = TcgaBrcaRaw()
#
# # Pooled test dataset
# mydataset_pooled = FedTcgaBrca(train=False, pooled=True)
#
# # Center 2 train dataset
# mydataset_local2= FedTcgaBrca(center=2, train=True, pooled=False)

import torch
from flamby.utils import evaluate_model_on_tests

# 2 lines of code to change to switch to another dataset
from flamby.datasets.fed_heart_disease import (
# from flamby.datasets.fed_tcga_brca import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    metric,
    NUM_CLIENTS,
    Optimizer,
)
# from flamby.datasets.fed_tcga_brca import FedTcgaBrca as FedDataset
from flamby.datasets.fed_heart_disease import FedHeartDisease as FedDataset

# Instantiation of local train set (and data loader)), baseline loss function, baseline model, default optimizer
# train_dataset = FedDataset(center=0, train=True, pooled=False)
train_dataset = FedDataset(train=True, pooled=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
lossfunc = BaselineLoss()
model = Baseline()
optimizer = Optimizer(model.parameters(), lr=LR)

# Traditional pytorch training loop
for epoch in range(0, NUM_EPOCHS_POOLED):
    for idx, (X, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(X)
        loss = lossfunc(outputs, y)
        loss.backward()
        optimizer.step()

# Evaluation
# Instantiation of a list of the local test sets
test_dataloaders = [
    torch.utils.data.DataLoader(
        FedDataset(center=i, train=False, pooled=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    for i in range(NUM_CLIENTS)
]
# Function performing the evaluation
dict_cindex = evaluate_model_on_tests(model, test_dataloaders, metric)
print(dict_cindex)

# import torch
# from flamby.utils import evaluate_model_on_tests
#
# # 2 lines of code to change to switch to another dataset
# from flamby.datasets.fed_heart_disease import (
# # from flamby.datasets.fed_tcga_brca import (
#     BATCH_SIZE,
#     LR,
#     NUM_EPOCHS_POOLED,
#     Baseline,
#     BaselineLoss,
#     metric,
#     NUM_CLIENTS,
#     get_nb_max_rounds
# )
# # from flamby.datasets.fed_tcga_brca import FedTcgaBrca as FedDataset
# from flamby.datasets.fed_heart_disease import FedHeartDisease as FedDataset
#
# # 1st line of code to change to switch to another strategy
# from flamby.strategies.fed_avg import FedAvg as strat
#
# # We loop on all the clients of the distributed dataset and instantiate associated data loaders
# train_dataloaders = [
#     torch.utils.data.DataLoader(
#         FedDataset(center = i, train = True, pooled = False),
#         batch_size = BATCH_SIZE,
#         shuffle = True,
#         num_workers = 0
#     )
#     for i in range(NUM_CLIENTS)
# ]
#
# lossfunc = BaselineLoss()
# m = Baseline()
#
# # Federated Learning loop
# # 2nd line of code to change to switch to another strategy (feed the FL strategy the right HPs)
# args = {
#     "training_dataloaders": train_dataloaders,
#     "model": m,
#     "loss": lossfunc,
#     "optimizer_class": torch.optim.SGD,
#     "learning_rate": LR / 10.0,
#     "num_updates": NUM_EPOCHS_POOLED,
#     # This helper function returns the number of rounds necessary to perform approximately as many
#     # epochs on each local dataset as with the pooled training
#     "nrounds": get_nb_max_rounds(NUM_EPOCHS_POOLED),
# }
# s = strat(**args)
# m = s.run()[0]
#
# # Evaluation
# # We only instantiate one test set in this particular case: the pooled one
# test_dataloaders = [
#     torch.utils.data.DataLoader(
#         FedDataset(train = False, pooled = True),
#         batch_size = BATCH_SIZE,
#         shuffle = False,
#         num_workers = 0,
#     )
# ]
# # test_dataloaders = [
# #     torch.utils.data.DataLoader(
# #         FedDataset(center=i, train=False, pooled=False),
# #         batch_size=BATCH_SIZE,
# #         shuffle=False,
# #         num_workers=0,
# #     )
# #     for i in range(NUM_CLIENTS)
# # ]
# dict_cindex = evaluate_model_on_tests(m, test_dataloaders, metric)
# print(dict_cindex)



# from flamby.datasets.fed_heart_disease import HeartDiseaseRaw, FedHeartDisease
# # Raw dataset
# mydataset_raw = HeartDiseaseRaw()
# # Pooled train dataset
# mydataset_pooled = FedHeartDisease(train=True, pooled=True)
# # Center 1 train dataset
# mydataset_local1= FedHeartDisease(center=1, train=True, pooled=False)