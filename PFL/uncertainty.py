import random
import sys

import torch.nn as nn
from torch.nn import functional as F
# from torchensemble.utils.logging import set_logger
# from torchensemble import VotingClassifier
import torch
from flamby.utils import evaluate_model_on_tests
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# loading data
# 2 lines of code to change to switch to another dataset
from flamby.datasets.fed_heart_disease import (
    BATCH_SIZE,
    LR, #learning rate
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    metric,
    NUM_CLIENTS,
    Optimizer,
    get_nb_max_rounds
)
# from flamby.datasets.fed_tcga_brca import FedTcgaBrca as FedDataset
from flamby.datasets.fed_heart_disease import FedHeartDisease as FedDataset

# Instantiation of local train set (and data loader)), baseline loss function, baseline model, default optimizer

lossfunc = BaselineLoss()


def get_pooled_train_loader():
    train_dataset = FedDataset(train=True, pooled=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # print(train_loader.__len__())
    # print(next(iter(train_loader)))
    return train_loader


def get_local_train_loader(center=0):
    train_dataset = FedDataset(center=center, train=True, pooled=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # print(train_loader.__len__())
    # print(next(iter(train_loader)))
    return train_loader


def get_pooled_test_loader():
    test_dataloader = torch.utils.data.DataLoader(
        FedDataset(train = False, pooled = True),
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = 0,
    )
    return test_dataloader


def global_test_dataset():
    return [
        torch.utils.data.DataLoader(
            FedDataset(train = False, pooled = True),
            batch_size = BATCH_SIZE,
            shuffle = False,
            num_workers = 0,
        )
    ]


def local_test_datasets():
    return [
        torch.utils.data.DataLoader(
            FedDataset(center=i, train=False, pooled=False),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        for i in range(NUM_CLIENTS)
    ]


def local_train_dataloaders():
    return [
        torch.utils.data.DataLoader(
            FedDataset(center = i, train = True, pooled = False),
            batch_size = BATCH_SIZE,
            shuffle = True,
            num_workers = 0
        )
        for i in range(NUM_CLIENTS)
    ]


test_loader = get_pooled_test_loader()
# print(test_loader.__len__())
# print(next(iter(test_loader)))

# logger = set_logger('heart_disease')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout') or m.__class__.__name__.startswith('MCDrop'):
            m.train()
            # print("Found dropout name:", m.__class__.__name__)
        # else: print(m.__class__.__name__)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set as {seed}")


def get_uncertainty_on_tests(
        model, test_dataloaders, use_gpu=True, return_pred=False, MCDO=False, T=1000, Ensemble=False, earlyexit=False, task="multiclass"
):
    entropy_dict = {}
    variance_dict = {}
    if Ensemble:
        for m in model:
            if torch.cuda.is_available() and use_gpu:
                m = m.cuda()
            m.eval()
    else:
        if torch.cuda.is_available() and use_gpu:
            model = model.cuda()
        model.eval()
    if MCDO:
        enable_dropout(model)
    
    # evaluating
    with torch.no_grad():
        for i in tqdm(range(len(test_dataloaders))):
            test_dataloader_iterator = iter(test_dataloaders[i])
            y_pred_final, y_true_final, entropy_final, variance_final = [], [], [], []
            for (X, y) in test_dataloader_iterator:
                if torch.cuda.is_available() and use_gpu:
                    X = X.cuda()
                    y = y.cuda()
                
                # calculating size of output
                if Ensemble:
                    out = model[0](X).detach().cpu()
                else:
                    out = model(X).detach().cpu()
              
                softmax = nn.Softmax(dim=-1) # for multiclass
                outputs = np.empty((0, list(out.size())[0], list(out.size())[1]))
                if MCDO:
                    set_seed(0)
                    for _ in range(T):
                        out = model(X).detach().cpu()
                        if task == "multiclass":
                            out = softmax(out)
                        outputs = np.vstack((outputs, out[np.newaxis, :, :])) 
                        # print(outputs)
                elif Ensemble:
                    for m in model:
                        # output += m(X) / float(len(model))
                        out = m(X).detach().cpu()
                        if task == "multiclass":
                            out = softmax(out)
                        outputs = np.vstack((outputs, out[np.newaxis, :, :])) # shape (forward_passes, n_samples, n_classes)
                elif earlyexit:
                    out = model(X).detach().cpu() # shape (n_samples, n_exits, n_classes)
                    # print(out[-1])
                    if task == "multiclass":
                        out = softmax(out).numpy()
                    else: 
                        out = out.numpy()
                    # print(out[-1])

                if MCDO or Ensemble:
                    # y_pred = output
                    y_pred = np.mean(outputs, axis=0) # shape (n_samples, n_classes)
                    variance = np.var(outputs, axis=0)
                    # print("variance:", variance)
                    var = np.mean(variance, axis=1)
                    # print("var:", var)
                    variance_final.append(var)
                elif earlyexit:
                    # print(out)
                    y_pred = np.mean(out, axis=1)
                    # print(y_pred.shape)
                    variance = np.var(out, axis=1)
                    var = np.mean(variance, axis=1)
                    variance_final.append(var)
                else:
                    out = model(X).detach().cpu()
                    if task == "multiclass":
                        y_pred = softmax(out).numpy()
                    else: 
                        y_pred = out.numpy()
                y = y.detach().cpu().numpy()
                y_pred_final.append(y_pred)
                y_true_final.append(y)

                epsilon = sys.float_info.min
                # Calculating entropy across multiple MCD forward passes
                entropy = -np.sum(y_pred *np.log(y_pred + epsilon), axis=-1) # shape (n_samples,)
                # print("entropy:", entropy)
                entropy_final.append(entropy)

            y_true_final = np.concatenate(y_true_final)
            y_pred_final = np.concatenate(y_pred_final)
            if MCDO or Ensemble or earlyexit:
                variance_final = np.concatenate(variance_final)
                variance_dict[f"client_test_{i}"] = variance_final
            entropy_final = np.concatenate(entropy_final)
            
            entropy_dict[f"client_test_{i}"] = entropy_final
    return variance_dict, entropy_dict


def evaluate_model_on_tests(
        model, test_dataloaders, metric, use_gpu=True, return_pred=False, MCDO=False, T=1000, Ensemble=False, earlyexit=False, task="multiclass"
):
    # initializing for evaluation
    results_dict = {}
    y_true_dict = {}
    y_pred_dict = {}
    entropy_dict = {}
    variance_dict = {}
    if Ensemble:
        for m in model:
            if torch.cuda.is_available() and use_gpu:
                m = m.cuda()
            m.eval()
    else:
        if torch.cuda.is_available() and use_gpu:
            model = model.cuda()
        model.eval()
    if MCDO:
        enable_dropout(model)
    
    # evaluating
    with torch.no_grad():
        for i in tqdm(range(len(test_dataloaders))):
            test_dataloader_iterator = iter(test_dataloaders[i])
            y_pred_final, y_true_final, entropy_final, variance_final = [], [], [], []
            for (X, y) in test_dataloader_iterator:
                if torch.cuda.is_available() and use_gpu:
                    X = X.cuda()
                    y = y.cuda()
                
                # calculating size of output
                if Ensemble:
                    out = model[0](X).detach().cpu()
                else:
                    out = model(X).detach().cpu()
                # print(list(out.size()))
                
                softmax = nn.Softmax(dim=-1) # for multiclass
                outputs = np.empty((0, list(out.size())[0], list(out.size())[1]))
                if MCDO:
                    set_seed(0)
                    for _ in range(T):
                        # output += model(X) / float(T)
                        # set_seed(_)
                        out = model(X).detach().cpu()
                        if task == "multiclass":
                            out = softmax(out)
                        outputs = np.vstack((outputs, out[np.newaxis, :, :])) # shape (forward_passes, n_samples, n_classes)
                        # print(outputs)
                elif Ensemble:
                    for m in model:
                        # output += m(X) / float(len(model))
                        out = m(X).detach().cpu()
                        if task == "multiclass":
                            out = softmax(out)
                        outputs = np.vstack((outputs, out[np.newaxis, :, :])) # shape (forward_passes, n_samples, n_classes)
                elif earlyexit:
                    out = model(X).detach().cpu() # shape (n_samples, n_exits, n_classes)
                    # print(out[-1])
                    if task == "multiclass":
                        out = softmax(out).numpy()
                    else: 
                        out = out.numpy()
                    # print(out[-1])

                if MCDO or Ensemble:
                    # y_pred = output
                    y_pred = np.mean(outputs, axis=0) # shape (n_samples, n_classes)
                    variance = np.var(outputs, axis=0)
                    # print("variance:", variance)
                    var = np.mean(variance, axis=1)
                    # print("var:", var)
                    variance_final.append(var)
                elif earlyexit:
                    # print(out)
                    y_pred = np.mean(out, axis=1)
                    # print(y_pred.shape)
                    variance = np.var(out, axis=1)
                    var = np.mean(variance, axis=1)
                    variance_final.append(var)
                else:
                    out = model(X).detach().cpu()
                    if task == "multiclass":
                        y_pred = softmax(out).numpy()
                    else: 
                        y_pred = out.numpy()
                y = y.detach().cpu().numpy()
                y_pred_final.append(y_pred)
                y_true_final.append(y)

                epsilon = sys.float_info.min
                # Calculating entropy across multiple MCD forward passes
                entropy = -np.sum(y_pred *np.log(y_pred + epsilon), axis=-1) # shape (n_samples,)
                # print("entropy:", entropy)
                entropy_final.append(entropy)

            y_true_final = np.concatenate(y_true_final)
            y_pred_final = np.concatenate(y_pred_final)
            if MCDO or Ensemble:
                variance_final = np.concatenate(variance_final)
                variance_dict[f"client_test_{i}"] = variance_final
            entropy_final = np.concatenate(entropy_final)
            results_dict[f"client_test_{i}"] = metric(y_true_final, y_pred_final)
            if return_pred:
                y_true_dict[f"client_test_{i}"] = y_true_final
                y_pred_dict[f"client_test_{i}"] = y_pred_final
                entropy_dict[f"client_test_{i}"] = entropy_final
    if return_pred:
        # print(variance_dict)
        return results_dict, y_true_dict, y_pred_dict, variance_dict, entropy_dict
    else:
        return results_dict


def plot_box(data, x_ticks, title=None, collection=True):
    # plt.style.use('ggplot')
    plt.style.use("seaborn-ticks")
    
    for i in range(len(x_ticks)):
        print(x_ticks[i], "=", data[i])
    
    # print(["client{i}" for i in range(8)])

    fig, ax = plt.subplots()
    if collection:
        fig, ax = plt.subplots(figsize=(16,9))
        ax.tick_params(axis='x', rotation=90)
    ax.boxplot(data)
    ax.set_xticklabels(x_ticks)
    ax.set_title(title)
    # ax.set_yscale("log")
    # plt.show()
    plt.savefig("figs/boxplot/" + title + ".png")
    plt.close()


def plot_hist(data, x_ticks, title=""):
    # print(x_ticks)
    import matplotlib as mpl
    mpl.rc('font', family='serif', serif='Times New Roman')
    sns.set(font_scale=2)
    plt.style.use("seaborn-ticks")

    for i in range(len(x_ticks)):
        print(x_ticks[i], "=", data[i])
    
    # kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)
    # for i in range(len(data)):
    #     plt.hist(data[i], **kwargs, label=x_ticks[i])

    kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2}, bins=30, norm_hist=True)
    for i in range(len(data)):
        sns.distplot(data[i], label=x_ticks[i], **kwargs)
        # sns.displot(data[i], label=x_ticks[i], kind="kde", fill=True)
    plt.gca().set(title='Probability Histogram', ylabel='Probability')
    plt.legend()
    sns.despine()
    # plt.show()
    plt.savefig("figs/hist/" + title + ".png")
    plt.close()


def plot_boxes(data, x_ticks, title=None):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(24,9))
    # fig, ax = plt.subplots()
    ax.boxplot(data)
    ax.set_xticklabels(x_ticks)
    ax.set_title(title)
    # ax.set_yscale("log")
    # ax.tick_params(axis='x', rotation=10)
    plt.show()
    # plt.savefig("figs/MCD-1000/" + title + ".png")


def cal_AUROC(predict, crt_idxs, wrg_idxs=None):
    # n_sample = min(len(crt_idxs), len(wrg_idxs)) // 2 + 1
    # random.seed(0)
    # crt_sample = random.sample(crt_idxs, n_sample)
    # wrg_sample = random.sample(wrg_idxs, n_sample)
    # target = torch.tensor([0 for i in crt_sample] + [1 for i in wrg_sample])
    # preds = torch.tensor([predict[i] for i in crt_sample] + [predict[i] for i in wrg_sample])
    if len(crt_idxs) == 0 or len(crt_idxs) == len(predict):
        print("Attention: 0 or 100 accuracy")
        print("calculated AUROC:", 1.0)
        return 1.0
    target = torch.tensor([0 if i in crt_idxs else 1 for i in range(len(predict))])
    preds = torch.tensor(predict)
    from torchmetrics.classification import BinaryAUROC
    metric = BinaryAUROC(thresholds=None)
    res = float(metric(preds, target))
    print("calculated AUROC:", res)
    return res


def selected_predict(uncertainty, y_true, y_pred, metric, percent=0.4):
    res = []
    n_data = []
    for k in y_true:
        # print(k)
        threshold = np.quantile(uncertainty[k], 1 - percent)
        # print(threshold)
        # print(list(uncertainty[k]))
        # print(y_true[k].shape)
        # print(y_pred[k].shape)
        y_true_masked = np.array([y for i, y in enumerate(y_true[k]) if np.isnan(uncertainty[k][i]) or uncertainty[k][i] <= threshold])
        y_pred_masked = np.array([y for i, y in enumerate(y_pred[k]) if np.isnan(uncertainty[k][i]) or uncertainty[k][i] <= threshold])
        # print(y_true_masked.shape)
        # print(y_pred_masked.shape)
        met = metric(y_true_masked, y_pred_masked)
        # print("selected prediction after dropping {} predictions:".format(percent), met)
        n_data.append(len(y_true[k]))
        res.append(met)
    if len(n_data) > 1:
        print(n_data)
        print(res)
        n_data.pop()
    return sum([res[i] * n for i, n in enumerate(n_data)]) / sum(n_data)


def evaluate(dict_cindex, y_true, y_pred, variance, entropy, uncertainty=True, id="", task="binary"):
    """
    only valid for binary classification now, due to the correct/wrong thing
    """
    data_variance, data_entropy = [], []
    xticks_variance, xticks_entropy = [], []
    print(dict_cindex)
    auroc_ent, auroc_var = None, None
    for k in y_true:
        print(k)
        correct_var, wrong_var, correct_ent, wrong_ent = [], [], [], []
        # for AUROC
        correct_idxs, wrong_idxs = [], []
        all_var, all_ent = [], []
        for i in range(len(y_true[k])):
            if task == "binary":
                correct = (y_pred[k][i] > 0.5) == y_true[k][i]
            # elif task == "multiclass":
            else:
                correct = (np.argmax(y_pred[k][i]) == y_true[k][i])
            # for AUROC
            all_ent.append(entropy[k][i])
            if uncertainty:
                all_var.append(variance[k][i])
            if correct:
                correct_idxs.append(i)
                if uncertainty:
                    correct_var.append(variance[k][i])
                correct_ent.append(entropy[k][i])
            else:
                wrong_idxs.append(i)
                if uncertainty:
                    wrong_var.append(variance[k][i])
                wrong_ent.append(entropy[k][i])
        
        data_entropy.extend([correct_ent, wrong_ent])
        xticks_entropy.extend(["C{}_CRT".format(k), "C{}_WRG".format(k)])
        print("Entropy")
        # print(np.mean(correct_ent))
        # print(np.mean(wrong_ent))
        auroc_ent = cal_AUROC(all_ent, correct_idxs, wrong_idxs)
        # plot_box([correct_ent, wrong_ent], ["correct", "wrong"], "model_{}_{}".format(str(id), k) + "_Entropy", collection=False)
        # plot_hist([correct_ent, wrong_ent], ["Correct", "Wrong"], "model_{}_{}".format(str(id), k) + "_Entropy")
        if uncertainty:
            data_variance.extend([correct_var, wrong_var])
            xticks_variance.extend(["{}_CRT".format(k), "{}_WRG".format(k)])
            print("Variance")
            # print(np.mean(correct_var))
            # print(np.mean(wrong_var))
            auroc_var = cal_AUROC(all_var, correct_idxs, wrong_idxs)
            # plot_box([correct_var, wrong_var], ["correct", "wrong"], "model_{}_{}".format(str(id), k) + "_variance", collection=False)
    # if uncertainty:
        # plot_box(data_variance, xticks_variance, "model_{}".format(str(id)) + "_variance")
    # plot_box(data_entropy, xticks_entropy, "model_{}".format(str(id)) + "_entropy")
    return data_entropy, data_variance, auroc_ent, auroc_var


def try_baseline(center=0):
    lossfunc = BaselineLoss()
    model = Baseline()
    optimizer = Optimizer(model.parameters(), lr=LR)
    # train_loader = get_local_train_loader(center=center)
    train_loader = get_pooled_train_loader()
    for epoch in range(0, NUM_EPOCHS_POOLED):
        for idx, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(X)
            loss = lossfunc(outputs, y)
            loss.backward()
            optimizer.step()
    # test_dataloader = get_pooled_test_loader()
    test_dataloaders = local_test_datasets() + global_test_dataset()
    dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(model, test_dataloaders, metric, MCDO=True, return_pred=True, T=1000)
    evaluate(dict_cindex, y_true, y_pred, variance, entropy, uncertainty=True, id="centralized")


def try_MC(T=100, center=0):
    # try out MC-dropout
    lossfunc = BaselineLoss()
    model = Baseline(MCDO=True)
    optimizer = Optimizer(model.parameters(), lr=LR)
    train_loader = get_pooled_train_loader()
    print("using dropout")

    for epoch in range(0, NUM_EPOCHS_POOLED):
        for idx, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(X)
            loss = lossfunc(outputs, y)
            loss.backward()
            optimizer.step()

    test_dataloaders = local_test_datasets() + global_test_dataset()
    dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(model, test_dataloaders, metric, return_pred=True, MCDO=True, T=T)
    evaluate(dict_cindex, y_true, y_pred, variance, entropy, id="centralized")


def try_MC_FL(T=100):
    # 1st line of code to change to switch to another strategy
    # from flamby.strategies.fed_avg import FedAvg as strat
    from flamby.strategies.fed_prox import FedProx as strat

    lossfunc = BaselineLoss()
    m = Baseline()

    # Federated Learning loop
    # 2nd line of code to change to switch to another strategy (feed the FL strategy the right HPs)
    args = {
        "training_dataloaders": local_train_dataloaders(),
        "model": m,
        "loss": lossfunc,
        "optimizer_class": torch.optim.SGD,
        # "optimizer_class": Optimizer,
        "learning_rate": LR / 10.0,
        "num_updates": 100,
        # This helper function returns the number of rounds necessary to perform approximately as many
        # epochs on each local dataset as with the pooled training
        "nrounds": get_nb_max_rounds(100),
        "mu": 0
    }
    s = strat(**args)
    m = s.run()[0]
    test_dataloaders = local_test_datasets() + global_test_dataset()
    dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(m, test_dataloaders, metric, return_pred=True, MCDO=True, T=T)
    evaluate(dict_cindex, y_true, y_pred, variance, entropy, id="fedavg")
    # # personalized
    # for id, test_loader in enumerate(test_dataloaders):
    #     dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(model, [test_loader], metric, return_pred=True, MCDO=True, T=100)
    #     evaluate(dict_cindex, y_true, y_pred, variance, entropy, id=id)


def try_ensemble(num_models=10, center=0):
    print("using ensemble")
    models = [Baseline().to(device) for _ in range(num_models)]
    lossfunc = BaselineLoss()

    for model in models:
        train_loader = get_local_train_loader(center=center)
        optimizer = Optimizer(model.parameters(), lr=LR)
        for epoch in range(0, NUM_EPOCHS_POOLED):
            for idx, (X, y) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(X)
                loss = lossfunc(outputs, y)
                loss.backward()
                optimizer.step()

    test_dataloaders = [
        torch.utils.data.DataLoader(
            FedDataset(center=center, train=False, pooled=False),
            batch_size = BATCH_SIZE,
            shuffle = False,
            num_workers = 0,
        ),
        torch.utils.data.DataLoader(
            FedDataset(train = False, pooled = True),
            batch_size = BATCH_SIZE,
            shuffle = False,
            num_workers = 0,
        )
    ]
    dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(models, test_dataloaders, metric, Ensemble=True, return_pred=True)
    evaluate(dict_cindex, y_true, y_pred, variance, entropy, id=center)


# def try_ensemble_old():
#     model = VotingClassifier(
#         estimator=Baseline(),
#         n_estimators=1,
#         cuda=False,
#     )

#     criterion = nn.BCELoss()
#     model.set_criterion(criterion)
#     train_loader = get_local_train_loader(center=0)
#     # model.set_criterion(BaselineLoss)

#     model.set_optimizer('Adam',  # parameter optimizer
#                         lr=LR,  # learning rate of the optimizer
#                         weight_decay=5e-4)  # weight decay of the optimizer

#     # Training
#     model.fit(train_loader=train_loader,  # training data
#               epochs=NUM_EPOCHS_POOLED)                 # the number of training epochs

#     # Evaluating
#     accuracy = model.predict(test_loader)
#     print(accuracy)


if __name__ == '__main__':
    # try_ensemble(1, 1)
    # try_ensemble(10, 1)
    # try_MC(False, 1)
    # try_MC(1000)
    # try_MC_FL(1000)
    try_baseline()
    # try_ensemble_old()
