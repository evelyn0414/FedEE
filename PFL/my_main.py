import collections
import random
from alg import algs
import os
import torch
import numpy as np
from tqdm import tqdm
import json
import importlib
import argparse

# dataset_name = "fed_ixi"
dataset_name = "fed_isic2019"
# dataset_name = "fed_har"
# dataset_name = "fed_pcg"
# dataset_name = "fed_heart_disease"

earlyexit = True
# earlyexit = False
N_exits = 5 # 4 if dataset_name == "fed_har" else 5

# importing
if dataset_name == "fed_har":
    from har_config import (
        BATCH_SIZE,
        LR,
        NUM_EPOCHS_POOLED, # NUM_EPOCHS_POOLED here is used as number of local iterations
        Baseline,
        BaselineLoss,
        BaselineEarlyExit,
        ExitWeightedLoss,
        metric,
        NUM_CLIENTS,
        get_nb_max_rounds,
        Optimizer
    )
    from har_config import (
        get_dataloader,
        NUM_EPOCHS_CENTRALIZED
    )
elif dataset_name == "fed_pcg":
    from audio_config import (
        BATCH_SIZE,
        LR,
        NUM_EPOCHS_POOLED, # NUM_EPOCHS_POOLED here is used as number of local iterations
        Baseline,
        BaselineLoss,
        BaselineEarlyExit,
        ExitWeightedLoss,
        metric,
        NUM_CLIENTS,
        get_nb_max_rounds,
        Optimizer
    )
    from audio_config import (
        get_dataloader,
        NUM_EPOCHS_CENTRALIZED
    )
elif dataset_name == "fed_isic2019":
    from flamby.datasets.fed_isic2019 import (
        BATCH_SIZE,
        LR,
        NUM_EPOCHS_POOLED, # NUM_EPOCHS_POOLED here is used as number of local iterations
        Baseline,
        BaselineLoss,
        BaselineEarlyExit,
        ExitWeightedLoss,
        metric,
        NUM_CLIENTS,
        get_nb_max_rounds,
        Optimizer
    )
    from flamby.datasets.fed_isic2019 import FedIsic2019 as FedDataset
    NUM_EPOCHS_CENTRALIZED = NUM_EPOCHS_POOLED
elif dataset_name == "fed_heart_disease":
    from flamby.datasets.fed_heart_disease import (
        BATCH_SIZE,
        LR,
        NUM_EPOCHS_POOLED, # NUM_EPOCHS_POOLED here is used as number of local iterations
        Baseline,
        BaselineLoss,
        # BaselineEarlyExit,
        # ExitWeightedLoss,
        metric,
        NUM_CLIENTS,
        get_nb_max_rounds,
        Optimizer
    )
    from flamby.datasets.fed_heart_disease import FedHeartDisease as FedDataset
    NUM_EPOCHS_CENTRALIZED = NUM_EPOCHS_POOLED
elif dataset_name == "fed_ixi":
    from flamby.datasets.fed_ixi import (
        BATCH_SIZE,
        LR,
        NUM_EPOCHS_POOLED, # NUM_EPOCHS_POOLED here is used as number of local iterations
        Baseline,
        BaselineLoss,
        # BaselineEarlyExit,
        # ExitWeightedLoss,
        metric,
        NUM_CLIENTS,
        get_nb_max_rounds,
        Optimizer
    )
    from flamby.datasets.fed_ixi import FedIXITiny as FedDataset
    NUM_EPOCHS_CENTRALIZED = NUM_EPOCHS_POOLED


if earlyexit:
    Baseline = BaselineEarlyExit
    BaselineLoss = ExitWeightedLoss

ROUND_PER_SAVE = 1

# from flamby.utils import evaluate_model_on_tests
from uncertainty import evaluate_model_on_tests
TASK = "binary" if dataset_name == "fed_pcg" else "multiclass"



def global_test_dataset():
    if dataset_name in ["fed_har", "fed_pcg"]:
        return [
            get_dataloader(train = False, pooled = True)
            ]
    else:
        return [
            torch.utils.data.DataLoader(
                FedDataset(train = False, pooled = True),
                batch_size = BATCH_SIZE,
                shuffle = False,
                num_workers = 0,
            )
        ]


def local_test_datasets():
    if dataset_name in ["fed_har", "fed_pcg"]:
        return [
            get_dataloader(center=i, train = False, pooled = False)
            for i in range(NUM_CLIENTS)
        ]
    else:
        return [
            torch.utils.data.DataLoader(
                FedDataset(center=i, train=False, pooled=False),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )
            for i in range(NUM_CLIENTS)
        ]


def train_dataset():
    if dataset_name in ["fed_har", "fed_pcg"]:
        return [
            get_dataloader(center=i, train = True, pooled = False)
            for i in range(NUM_CLIENTS)
        ]
    else:
        return [
            torch.utils.data.DataLoader(
                FedDataset(center = i, train = True, pooled = False),
                batch_size = BATCH_SIZE,
                shuffle = True,
                num_workers = 0,
                drop_last=True,
            )
            for i in range(NUM_CLIENTS)
        ]


def get_pooled_train_loader():
    if dataset_name in ["fed_har", "fed_pcg"]:
        train_loader = get_dataloader(train = True, pooled = True)
    else:
        train_dataset = FedDataset(train=True, pooled=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return [train_loader]


def initialize_args(alg="fedavg", device="cuda" if torch.cuda.is_available() else "cpu", centralized=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default=alg,
                        help='Algorithm to choose: [base | fedavg | fedbn | fedprox | fedap | metafed | fedee ]')
    parser.add_argument('--dataset', type=str, default=dataset_name,
                        help='Dataset to choose: [fed-heart-disease | fed-isic2019 | fed_camelyon16]')
    parser.add_argument('--save_path', type=str,
                        default='./cks/', help='path to save the checkpoint')
    parser.add_argument('--device', type=str,
                        default=device, help='[cuda | cpu]')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--num', type=str, default="0", help="number")
    if centralized:
        parser.add_argument('--iters', type=int, default=1, # 300
                            help='iterations for communication')
        parser.add_argument('--n_clients', type=int,
                            default=1, help='number of clients')
        parser.add_argument('--wk_iters', type=int, default=NUM_EPOCHS_CENTRALIZED, #changed from 1
                        help='optimization iters in local worker between communication')
    elif alg == "base":
        parser.add_argument('--iters', type=int, default=1, # 300
                            help='iterations for communication')
        parser.add_argument('--n_clients', type=int,
                            default=NUM_CLIENTS, help='number of clients')
        parser.add_argument('--wk_iters', type=int, default=NUM_EPOCHS_CENTRALIZED, #changed from 1
                        help='optimization iters in local worker between communication')
    else:
        parser.add_argument('--iters', type=int, default=get_nb_max_rounds(NUM_EPOCHS_POOLED), # 300
                                help='iterations for communication')
        parser.add_argument('--n_clients', type=int,
                            default=NUM_CLIENTS, help='number of clients')
        parser.add_argument('--wk_iters', type=int, default=NUM_EPOCHS_POOLED, #changed from 1
                        help='optimization iters in local worker between communication')
    parser.add_argument('--lr', type=float, default=LR, help='learning rate')

    
    parser.add_argument('--nosharebn', action='store_true', default=True,
                        help='not share bn')

    parser.add_argument('--plan', type=int,
                        default=1, help='choose the feature type')
    parser.add_argument('--pretrained_iters', type=int,
                        default=150, help='iterations for pretrained models')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # algorithm-specific parameters
    parser.add_argument('--mu', type=float, default=1e-2,
                        help='The hyper parameter for fedprox')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='threshold to use copy or distillation, hyperparmeter for metafed')
    parser.add_argument('--lam', type=float, default=1.0,
                        help='init lam, hyperparmeter for metafed')
    parser.add_argument('--model_momentum', type=float,
                        default=0.5, help='hyperparameter for fedap')
    args = parser.parse_args()
    return args



def train(strategy="fedavg", device="cuda" if torch.cuda.is_available() else "cpu", mode="", num="", pretrain=False, measure_one_round=False):
    import time
    SAVE_PATH = os.path.join('./cks/'+ mode, dataset_name + "_" + strategy + num)
    SAVE_PATH_RDS = os.path.join('./cks/'+ mode, dataset_name + "_" + strategy + num)
    SAVE_LOG = os.path.join('./cks/' + mode, "log_" + dataset_name + "_" + strategy + num)
    SAVE_LOG_RDS = os.path.join('./cks/' + mode, "log_" + dataset_name + "_" + strategy + num)
    test_loaders = local_test_datasets() + global_test_dataset()
    if strategy == "pooled":
        train_loaders = get_pooled_train_loader()
        val_loaders = global_test_dataset()
        args = initialize_args(strategy, device, centralized=True)
        algclass = algs.get_algorithm_class("base")(args, Baseline(), BaselineLoss(), Optimizer)
    else:
        train_loaders = train_dataset()
        val_loaders = local_test_datasets()
        args = initialize_args(strategy, device)
        algclass = algs.get_algorithm_class(strategy)(args, Baseline(), BaselineLoss(), Optimizer)
        args = initialize_args(strategy, device)
    if strategy == "metafed":
        algclass = algs.get_algorithm_class(strategy)(args, Baseline(), BaselineLoss(), metric=metric)
    start_iter, n_rounds = 0, args.iters
    wk_iters = args.wk_iters
    logs, client_logs = [],  [[] for _ in range(args.n_clients)]
    loaded_from_cks = False
    
    print("strategy:", strategy, "device:", device)
    if os.path.exists(SAVE_PATH_RDS):
        logs, client_logs = load_model(SAVE_PATH_RDS, algclass, device=device, fedap=("fedap" in strategy))
        start_iter = len(logs)
        print(f"============ Loaded model from train round {start_iter} from RDS ============")
        loaded_from_cks = True
    elif os.path.exists(SAVE_PATH):
        logs, client_logs = load_model(SAVE_PATH, algclass, device=device, fedap=("fedap" in strategy))
        start_iter = len(logs)
        print(f"============ Loaded model from train round {start_iter} from home dir ============")
        # for i, l in enumerate(logs):
            # print("round", i)
            # print(l)
        # print(client_logs)
        loaded_from_cks = True
    elif pretrain: print("ERROR: pretrain model not found")
    
    if pretrain: return algclass

    if strategy in ['fedap'] and not loaded_from_cks:
        algclass.set_client_weight(train_loaders)
    elif args.alg == 'metafed':
        algclass.init_model_flag(train_loaders, val_loaders)
        args.iters = args.iters-1
        print('Common knowledge accumulation stage')

    # res = evaluate_model_on_tests(algclass.server_model, test_loaders, metric)
    # print("before training", res)

    for a_iter in range(start_iter, n_rounds):
        print(f"============ Train round {a_iter} ============")
        ts = time.time()
        print("starting clock time", ts)

        if strategy == 'metafed':
            for c_idx in range(args.n_clients):
                algclass.client_train(
                    c_idx, train_loaders[algclass.csort[c_idx]], a_iter)
            algclass.update_flag(val_loaders)
        else:
            # local client training
            for wi in tqdm(range(wk_iters)):
                for client_idx in range(args.n_clients):
                    algclass.client_train(client_idx, train_loaders[client_idx], a_iter)

            # server aggregation
            algclass.server_aggre()
        ts2 = time.time()
        print("ending clock time", ts2, "wall clock time", ts2 - ts)
        if measure_one_round:
            return

        res = evaluate_model_on_tests(algclass.server_model, test_loaders, metric, earlyexit=earlyexit, task=TASK)
        logs.append(res)
        print("performance of server model", res)
        for i, tmodel in enumerate(algclass.client_model):
            client_res = evaluate_model_on_tests(tmodel, [test_loaders[i]], metric, earlyexit=earlyexit, task=TASK)
            performance = client_res["client_test_0"]
            print(f"result for client model {i}:", performance)
            client_logs[i].append(performance)
            if strategy == "pooled":
                print(evaluate_model_on_tests(tmodel, test_loaders, metric, earlyexit=earlyexit, task=TASK))

        if a_iter % ROUND_PER_SAVE == 0:
            print(f' Saving the local and server checkpoint to {SAVE_PATH_RDS}')
            tosave = {'current_epoch': a_iter, 'current_metric': res[f"client_test_{args.n_clients}"], 'logs': np.array(logs), "client_logs": np.array(client_logs)}
            for i,tmodel in enumerate(algclass.client_model):
                tosave['client_model_'+str(i)]=tmodel.state_dict()
            tosave['server_model']=algclass.server_model.state_dict()
            if "fedap" in args.alg:
                tosave["client_weight"] = algclass.client_weight
            torch.save(tosave, SAVE_PATH_RDS)
            tosave_log = {'server_logs': logs, 'client_logs': client_logs}
            torch.save(tosave_log, SAVE_LOG_RDS)

    if args.alg == 'metafed':
        print('Personalization stage')
        for c_idx in range(args.n_clients):
            algclass.personalization(
                c_idx, train_loaders[algclass.csort[c_idx]], val_loaders[algclass.csort[c_idx]])
            res = evaluate_model_on_tests(algclass.client_model[c_idx], [test_loaders[c_idx]], metric, task=TASK, earlyexit=earlyexit)
            print(f"final result for client {c_idx}:", res["client_test_0"])

    # print(logs)
    # print(client_logs)
    return algclass


def load_model(SAVE_PATH, algclass, device="cpu", fedap=False):
    print(f"============ Loading model ============")
    loaded = torch.load(SAVE_PATH, map_location=torch.device(device))
    algclass.server_model.load_state_dict(loaded["server_model"])
    # test_loaders = global_test_dataset()
    # res = evaluate_model_on_tests(algclass.server_model, test_loaders, metric)
    # print("server performance:", res["client_test_0"])
    test_loaders = local_test_datasets()
    model_bns = []
    for i, tmodel in enumerate(algclass.client_model):
        tmodel.load_state_dict(loaded['client_model_'+str(i)])
        # res = evaluate_model_on_tests(tmodel, [test_loaders[i]], metric, task=TASK, earlyexit=earlyexit)
        # print(f"performance for client {i}:", res["client_test_0"])
        # print("bn.running_mean and bn.running_var", [bn.running_mean for bn in tmodel.bns], [bn.running_var for bn in tmodel.bns])
        # bns = [np.concatenate([bn.running_mean.cpu().numpy(), bn.running_var.cpu().numpy()]) for bn in tmodel.bns]
        # print(bns)
        # model_bns.append(np.concatenate(bns))
        # print(model_bns)
    # model_bns = np.array(model_bns)
    # compute_bn_similarity(model_bns)
    if fedap:
        if "client_weight" in loaded:
            algclass.client_weight = loaded["client_weight"]
            print(algclass.client_weight)
        else:
            print("Warning: fedAP weight not loaded. Okay with testing, not okay for training.")
            # print("Re-calculating.")
            # algclass.set_client_weight(train_dataset())
    return loaded["logs"].tolist(), loaded["client_logs"].tolist()


def load_log(SAVE_LOG):
    loaded = torch.load(SAVE_LOG, map_location=torch.device('cpu'))
    logs = loaded["server_logs"]
    client_logs = loaded["client_logs"]
    return logs, client_logs


def calculate_stable_performance(logs, client_logs):
    window_size = 3
    client_num = len(client_logs)
    client_res = []
    server_res = np.mean([log[f"client_test_{client_num}"] for log in logs[-window_size:]])
    for client in range(client_num):
        client_res.append(np.mean(client_logs[client][-window_size:]))
    return [server_res] + client_res
    # return server_res, client_res


def get_res_from_log(strategy, mode=""):
    print(dataset_name, strategy, mode)
    SAVE_LOG = os.path.join('./cks/'+ mode, "log_" + dataset_name + "_" + strategy)
    logs, client_logs = load_log(SAVE_LOG)
    if strategy not in ["base", "fedbn", "fedap"]:
        print(strategy + "_server = ", [log[f"client_test_{NUM_CLIENTS}"] for log in logs])
    n_rounds = len(client_logs[0])
    print( strategy + "_local = ", [np.mean([client_logs[client][i] for client in range(NUM_CLIENTS)]) for i in range(n_rounds)])
    
    # res = calculate_stable_performance(logs, client_logs)
    # for c in res:
    #     print(c, end="\t")
    # print(np.mean(res[1:]), end="\t")
    # print(np.std(res[1:]), end="\t")
    # print("")


def MCD_evaluation(strategy="fedavg", device="cuda" if torch.cuda.is_available() else "cpu", uncertainty=False, T=1000, mode="", num=""):
    print("============ Evaluating strategy {} ============".format(strategy))
    from uncertainty import evaluate_model_on_tests, evaluate, plot_box, selected_predict
    # SAVE_PATH = os.path.join('./cks/' + mode, dataset_name + "_" + strategy)
    # SAVE_LOG = os.path.join('./cks/' + mode, "log_" + dataset_name + "_" + strategy)

    # if strategy == "pooled":
    #     from uncertainty import get_pooled_train_loader
    #     args = initialize_args(strategy, device, centralized=True)
    #     algclass = algs.get_algorithm_class("base")(args, Baseline(), BaselineLoss(), Optimizer)
    # else:
    #     args = initialize_args(strategy, device)
    #     algclass = algs.get_algorithm_class(strategy)(args, Baseline(), BaselineLoss(), Optimizer)

    # if os.path.exists(SAVE_PATH):
    #     logs, client_logs = load_model(SAVE_PATH, algclass)
    #     # print("============ loaded model============ ")
    #     # print(logs)
    #     # print(client_logs)
    # else:
    #     print("ERROR:model not found")
    #     return
    
    algclass = train(strategy, mode="Ensemble/" + mode, num=num, pretrain=True)
    
    # test_loaders = local_test_datasets() + global_test_dataset()
    

    print(f"============ MC-Dropout Test {strategy} ============")
    
    # metric_matrix = []
    # data_entropy, xticks, data_variance = [], [], []
    # auroc_ents, auroc_vars, sp_ents, sp_vars, n_data = [], [], [], [], []
    # for i in range(NUM_CLIENTS):
    #     print("============ MCD Test model {} ============".format(i))
    #     n_data.append(test_loaders[i].dataset.__len__())
    #     if strategy == "pooled":
    #         tmodel = algclass.client_model[0]
    #     else:
    #         tmodel = algclass.client_model[i]
    #     dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(tmodel, [test_loaders[i]], metric, MCDO=uncertainty, T=T, return_pred=True)
    #     sp_ents.append(selected_predict(entropy, y_true, y_pred, metric))
    #     if uncertainty:
    #         sp_vars.append(selected_predict(variance, y_true, y_pred, metric))
    #     ent, var, auroc_ent, auroc_var = evaluate(dict_cindex, y_true, y_pred, variance, entropy, id=strategy + "_model_"+str(i), uncertainty=uncertainty, task=TASK)
    #     data_entropy.extend(ent)
    #     data_variance.extend(var)
    #     auroc_ents.append(auroc_ent)
    #     auroc_vars.append(auroc_var)
    #     metric_matrix.append(dict_cindex["client_test_0"])
    #     xticks.extend(["C{}_CRT".format(i), "C{}_WRG".format(i)])
    # # plot_box(data_entropy, xticks, strategy + "_entropy" )

    # print(n_data)
    # print("weighted average performance", sum([metric_matrix[i] * n for i, n in enumerate(n_data)]) / sum(n_data))
    # final_auroc_ent = sum([auroc_ents[i] * n for i, n in enumerate(n_data)]) / sum(n_data)
    # print("Ent. AUROC", final_auroc_ent)
    # if uncertainty:
    #     plot_box(data_variance, xticks, strategy + "_variance")
    #     final_auroc_var = sum([auroc_vars[i] * n for i, n in enumerate(n_data)]) / sum(n_data)
    #     print("Var. AUROC", final_auroc_var)
    # else: final_auroc_var = None
    
    if strategy in ["pooled", "fedavg", "fedprox", "fedprox2"]:
        if strategy == "pooled":
            algclass.server_model = algclass.client_model[0]
        print("===== test/centralized server model on pooled test set ======")
        dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(algclass.server_model, global_test_dataset(), metric, MCDO=uncertainty, T=T, return_pred=True, task=TASK, earlyexit=earlyexit)
        sp_ent = selected_predict(entropy, y_true, y_pred, metric)
        print("Weighted selective prediction res based on entropy", sp_ent)
        if uncertainty:
            print("Selective prediction res based on variance", selected_predict(variance, y_true, y_pred, metric))
        data_entropy, data_variance, final_auroc_ent, final_auroc_var = evaluate(dict_cindex, y_true, y_pred, variance, entropy, id= "centralized_dataset_"+ dataset_name + strategy + ("_earlyexit" if earlyexit else "") + ("_dropout" if uncertainty else ""), uncertainty=uncertainty, task=TASK)
    # return final_auroc_ent, final_auroc_var
    
        res_names = "accs, sp_ents, auroc_ents".split(", ")
        return res_names, [dict_cindex["client_test_0"], sp_ent, final_auroc_ent]


def ensemble(strategy="fedavg", device="cuda" if torch.cuda.is_available() else "cpu", num_models=5, ood=False, seed=None, mode=""):
    from uncertainty import evaluate_model_on_tests, evaluate, plot_box, selected_predict
    print("using ensemble")
    ensembles = []
    # if strategy == "pooled":
    #     args = initialize_args(strategy, device, centralized=True)
    # else:
    #     args = initialize_args(strategy, device)
    
    if seed is not None:
        set_seed(seed)
        if dataset_name == "fed_isic2019":
            model_list = [i for i in range(5) if i != seed]
        else:
            model_list = random.sample(range(0, 10), 5)
    else:
        model_list = range(num_models)
    for i in model_list:
        model = train(strategy, mode="Ensemble/" + mode, num=str(i), pretrain=True)
        ensembles.append(model)

    # test_loaders = local_test_datasets() + global_test_dataset()
    test_loaders = global_test_dataset()
    print(f"============ Ensemble Test {strategy} ============")
    # metric_matrix = []
    # data_entropy, xticks, data_variance = [], [], []
    # auroc_ents, auroc_vars, sp_ents, sp_vars, n_data = [], [], [], [], []
    # for i in range(NUM_CLIENTS):
    #     print("============ Ensemble Test model {} ============".format(i))
    #     # n_data.append(test_loaders[i].__len__())
    #     n_data.append(test_loaders[i].dataset.__len__())
    #     if strategy == "pooled":
    #         tmodel = [algclass.client_model[0] for algclass in ensembles]
    #     else:
    #         tmodel = [algclass.client_model[i] for algclass in ensembles]
    #     if ood:
    #         met, ent, var, auroc_ent, auroc_var = [], [], [], [], []
    #         for j in range(args.n_clients):
    #             # if j == i: continue
    #             dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(tmodel, [test_loaders[j]], metric, Ensemble=True, return_pred=True, task=TASK, earlyexit=earlyexit)
    #             ent_single, var_single, auroc_ent_single, auroc_var_single = evaluate(dict_cindex, y_true, y_pred, variance, entropy, id="ensemble_model_{}_{}_data_{}".format(strategy, str(i), str(j)), uncertainty=True, task=TASK)
    #             ent.append(np.mean(entropy["client_test_0"])); var.append(np.mean(variance["client_test_0"]))
    #             auroc_ent.append(auroc_ent_single); auroc_var.append(auroc_var_single)
    #             met.append(dict_cindex["client_test_0"])
    #             data_entropy.append(ent)
    #             data_variance.append(var)
    #     else:
    #         dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(tmodel, [test_loaders[i]], metric, Ensemble=True, return_pred=True, task=TASK, earlyexit=earlyexit)
    #         sp_ents.append(selected_predict(entropy, y_true, y_pred, metric))
    #         sp_vars.append(selected_predict(variance, y_true, y_pred, metric))
    #         ent, var, auroc_ent, auroc_var = evaluate(dict_cindex, y_true, y_pred, variance, entropy, id="ensemble_model_{}_{}".format(strategy, str(i)), uncertainty=True, task=TASK)
    #         met = dict_cindex["client_test_0"]
    #         data_entropy.extend(ent)
    #         data_variance.extend(var)
    #     # data_entropy.append(ent)
    #     # data_variance.append(var)
    #     auroc_ents.append(auroc_ent)
    #     auroc_vars.append(auroc_var)
    #     metric_matrix.append(met)
    #     # xticks.extend(["C{}_CRT".format(i), "C{}_WRG".format(i)])
    #     xticks.extend(["Correct" + "{" + str(i+1) + "}", "Wrong" + "{" + str(i+1) + "}"])
    # if ood:
    #     return np.array(metric_matrix), np.array(data_entropy), np.array(data_variance), np.array(auroc_ents), np.array(auroc_vars)
    # else:
    #     # data_entropy.extend(ent)
    #     # data_variance.extend(var)
    #     print(n_data)
    #     print("weighted average performance", sum([metric_matrix[i] * n for i, n in enumerate(n_data)]) / sum(n_data))
    #     # plot_box(data_entropy, xticks, dataset_name + "ensemble_{}_entropy".format(strategy))
    #     final_auroc_ent = sum([auroc_ents[i] * n for i, n in enumerate(n_data)]) / sum(n_data)
    #     print("Ent. AUROC", final_auroc_ent)
    #     # plot_box(data_variance, xticks, "ensemble_{}_variance".format(strategy))
    #     final_auroc_var = sum([auroc_vars[i] * n for i, n in enumerate(n_data)]) / sum(n_data)
    #     print("Var. AUROC", final_auroc_var)
    if strategy in ["pooled", "fedavg", "fedprox", "fedprox2"]:
        if strategy == "pooled":
            models = [algclass.client_model[0] for algclass in ensembles]
        else:
            models = [algclass.server_model for algclass in ensembles]
        print("===== test server/centralized model ======")
        dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(models, test_loaders, metric, Ensemble=True, return_pred=True, task=TASK, earlyexit=earlyexit)
        sp_ent = selected_predict(entropy, y_true, y_pred, metric)
        print("Weighted selective prediction res based on entropy", sp_ent)
        sp_var = selected_predict(variance, y_true, y_pred, metric)
        print("Weighted selective prediction res based on variance", sp_var)
        data_entropy, data_variance, final_auroc_ent, final_auroc_var = evaluate(dict_cindex, y_true, y_pred, variance, entropy, id="ensemble_model_" + strategy, task=TASK)
        
        res_names = "accs, sp_ents, auroc_ents".split(", ")
        return res_names, [dict_cindex["client_test_0"], sp_ent, final_auroc_ent]



def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")


def get_weighted_avg(nums, weights, eta=np.e):
    if len(nums) != len(weights):
        print("Error: len of nums and weights don't match")
    if eta < 1 or eta > np.e:
        print("Error: eta out of range")
    if eta != np.e:
        for i in range(len(weights)):
            weights[i] = eta ** np.log(weights[i])
    return sum([nums[i] * n for i, n in enumerate(weights)]) / sum(weights)


import seaborn as sns
import matplotlib.pyplot as plt


def heatmap(data, ax, normalization=True):
    print(data)
    if normalization:
        data = data.T
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        data = data.T
    # print(normed.T)
    sns.heatmap(data, ax=ax, annot=True,  fmt=".2f", cmap="crest", annot_kws={"size":4})


def cross_clients(alg):
    plt.figure(figsize=(45, 30))
    matrixs = [[] for i in range(5)]
    for seed in range(5):
        set_seed(seed)
        mtxs = ensemble(alg, ood=True)
        for i, mt in enumerate(mtxs):
            matrixs[i].append(mt)
    for i, mt in enumerate(matrixs):
        if i in [1, 2]:
            heatmap(np.mean(mt, axis=0), ax=plt.subplot(2, 3, i+1))
        else:
            heatmap(np.mean(mt, axis=0), ax=plt.subplot(2, 3, i+1), normalization=False)
    plt.savefig("figs/matrix/ensemble_" + alg + ".png")
    plt.close()


def compute_bn_similarity(bns):
    from sklearn.metrics import pairwise_distances
    from sklearn.metrics.pairwise import cosine_similarity
    # print(bns)
    # print(pairwise_distances(bns))
    print(cosine_similarity(bns))


def fine_tune_and_test(model, train_dataloader, test_dataloaders, device, MCDO=False, Ensemble=False, FT=5, lr=LR, st=""):
    uncertainty = MCDO or Ensemble
    import copy
    from uncertainty import evaluate_model_on_tests, evaluate, selected_predict
    lossfunc = BaselineLoss()
    origin_model = copy.deepcopy(model)

    from util.traineval import train, train_prox

    class prox_arg:
        def __init__(self, mu):
            self.mu = mu
    set_seed(0)
    if dataset_name == "fed_isic2019": lr = 0.0001
    print("fine-tune lr:", lr)
    if Ensemble:
        for m in model:
            optimizer = Optimizer(m.parameters(), lr=lr)
            for epoch in range(0, FT):
                train(m, train_dataloader, optimizer, lossfunc, device)
    else:
        optimizer = Optimizer(model.parameters(), lr=lr) # 0.0001 for pooled?
        for epoch in range(0, FT):
            train(model, train_dataloader, optimizer, lossfunc, device)
        # w_diff = torch.tensor(0., device=device)
        # for w, w_t in zip(origin_model.parameters(), model.parameters()):
        #     w_diff += torch.pow(torch.norm(w - w_t), 2)
        # print(torch.sqrt(w_diff).detach().cpu().numpy())
        # return [np.concatenate([bn.running_mean.cpu().numpy(), bn.running_var.cpu().numpy()]) for bn in model.bns]
        
        # train_prox(prox_arg(1e-3), model, global_model, train_dataloader, optimizer, lossfunc, device)
    dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(model, test_dataloaders, metric, return_pred=True, MCDO=MCDO, Ensemble=Ensemble, task=TASK, earlyexit=earlyexit, T=N_exits) #, T=4) #T=1000
    # print("fine-tune epoch ", epoch)
    # print(dict_cindex)
    sp_ent = selected_predict(entropy, y_true, y_pred, metric)
    sp_var = None
    if uncertainty:
        sp_var = selected_predict(variance, y_true, y_pred, metric)
        print(sp_var)
    print(sp_ent, sp_var)
    ent, var, auroc_ent, auroc_var = evaluate(dict_cindex, y_true, y_pred, variance, entropy, uncertainty=uncertainty, task=TASK, id="_".join([dataset_name, st]))
    print(auroc_ent, auroc_var)
    return dict_cindex['client_test_0'], sp_ent, sp_var, auroc_ent, auroc_var, ent


def fine_tune_testing(strategy, device="cuda" if torch.cuda.is_available() else "cpu", MCDO=False, FT=5, mode='', Ensemble=False, lr=LR, num=""):
    uncertainty = MCDO or Ensemble
    method = "base"
    if MCDO: method = "Dropout"
    elif Ensemble: method = "Ensemble"
    elif earlyexit: method = "earlyexit"
    f = "-FT" if FT > 0 else ""

    if dataset_name == "fed_isic2019" and earlyexit: lr = lr / 2

    print("============ Evaluating strategy {} with fine-tuning {} iters, Dropout {} Ensemble {} ============".format(strategy, FT, MCDO, Ensemble))
    from uncertainty import evaluate_model_on_tests, evaluate, plot_box, selected_predict
    # SAVE_PATH = os.path.join('./cks/' + mode, dataset_name + "_" + strategy)
    # SAVE_LOG = os.path.join('./cks/' + mode, "log_" + dataset_name + "_" + strategy)

    if strategy == "pooled":
        from uncertainty import get_pooled_train_loader
        args = initialize_args(strategy, device, centralized=True)
        algclass = algs.get_algorithm_class("base")(args, Baseline(), BaselineLoss(), Optimizer)
        lr = 0.0001
    else:
        args = initialize_args(strategy, device)
        algclass = algs.get_algorithm_class(strategy)(args, Baseline(), BaselineLoss(), Optimizer)

    ensembles = []
    if Ensemble:
        for i in range(5):
            model = train(strategy, mode="Ensemble/", num=str(i), pretrain=True)
            ensembles.append(model)
    else:
        # if os.path.exists(SAVE_PATH):
        #     load_model(SAVE_PATH, algclass, device=device)
        # else:
        #     print("ERROR:model not found")
        algclass = train(strategy, mode=mode, pretrain=True, num=num)

    train_loaders = train_dataset()
    test_loaders = local_test_datasets() + global_test_dataset()
    
    sp_ent_arrs = []
    model_bns = []
    data_entropy, xticks = [], []
    n_data, accs, sp_ents, sp_vars, auroc_ents, auroc_vars = [], [], [], [], [], []
    res_list = [accs, sp_ents, sp_vars, auroc_ents, auroc_vars]
    res_names = "accs, sp_ents, sp_vars, auroc_ents, auroc_vars".split(", ")
    for client_idx in range(NUM_CLIENTS):
        n_data.append(test_loaders[client_idx].dataset.__len__())
        if Ensemble and strategy == "pooled":
            model = [algclass.client_model[0] for algclass in ensembles]
        elif Ensemble:
            model = [algclass.client_model[client_idx] for algclass in ensembles]
        elif strategy == "pooled":
            model = algclass.client_model[0]
        else:
            model = algclass.client_model[client_idx]
        # print("client", client_idx)
        # bns = fine_tune_and_test(model, train_loaders[client_idx], [test_loaders[client_idx]], device, MCDO=MCDO, Ensemble=Ensemble, FT=FT)
        # model_bns.append(np.concatenate(bns))
        acc, sp_ent, sp_var, auroc_ent, auroc_var, ent = fine_tune_and_test(model, train_loaders[client_idx], [test_loaders[client_idx]], device, MCDO=MCDO, Ensemble=Ensemble, FT=FT, st=strategy, lr=lr)
        res = [acc, sp_ent, sp_var, auroc_ent, auroc_var]
        for i, val in enumerate(res):
            res_list[i].append(val)
        data_entropy.extend(ent)
        xticks.extend(["Correct" + "{" + str(client_idx +1) + "}", "Wrong" + "{" + str(client_idx +1) + "}"])
        # evolution of selective prediction
        # sp_ent_array = plot_selective_prediction_range(model, train_loaders[client_idx],
        #                                                     [test_loaders[client_idx]], device, MCDO=MCDO,
        #                                                     Ensemble=Ensemble, FT=FT)
        # sp_ent_arrs.append(sp_ent_array)

    for i, res in enumerate(res_list):
        if "var" in res_names[i] and not uncertainty:
            continue
        print(res_names[i])
        print(res)
        print("weighted average", get_weighted_avg(res, n_data))
    # model_bns = np.array(model_bns)
    # compute_bn_similarity(model_bns)
    # plot_box(data_entropy, xticks, dataset_name + strategy + f + method)

    # arr = []
    # for i in range(len(sp_ent_arrs[0])):
    #     arr.append(get_weighted_avg([sp_ent_arrs[c][i] for c in range(NUM_CLIENTS)], n_data))
    # print(method + strategy + f, "=", arr)


def fine_tune_testing_5run(strategy, device="cuda" if torch.cuda.is_available() else "cpu", MCDO=False, FT=5, mode='', Ensemble=False, save=True, eta=np.e, overwrite=False):
    uncertainty = MCDO or Ensemble
    method = "base"
    if MCDO: method = "Dropout"
    elif Ensemble: method = "Ensemble"
    elif earlyexit: method = "fedee" #"earlyexit"
    f = "-FT" if FT > 0 else ""

    file_name = dataset_name + "_average.json" if eta == 1 else dataset_name + ".json"
    print(file_name)
    
    if os.path.exists(file_name):
        res_dict = json.loads(open(file_name).readline())
        print(res_dict)
    else:
        res_dict = {}
    print("_".join([strategy+f, method, "auroc_ents"]))
    
    if "_".join([strategy+f, method, "auroc_ents"]) in res_dict:
        print("strategy {} with fine-tuning {} iters, {} already run".format(strategy, FT, method))
        if save and not overwrite: 
            # don't quit if not saving, re-running deliverately
            return
    
    print("============ Evaluating strategy {} with fine-tuning {} iters, {} ============".format(strategy, FT, method))
    from uncertainty import evaluate_model_on_tests, evaluate, plot_box, selected_predict
    # SAVE_PATH = os.path.join('./cks/' + mode, dataset_name + "_" + strategy)
    # SAVE_LOG = os.path.join('./cks/' + mode, "log_" + dataset_name + "_" + strategy)

    # if strategy == "pooled":
    #     from uncertainty import get_pooled_train_loader
    #     args = initialize_args(strategy, device, centralized=True)
    #     algclass = algs.get_algorithm_class("base")(args, Baseline(), BaselineLoss(), Optimizer)
    # else:
    #     args = initialize_args(strategy, device)
    #     algclass = algs.get_algorithm_class(strategy)(args, Baseline(), BaselineLoss(), Optimizer)
    
    
    def fine_tune_test_one_run(t=0):
        ensembles = []
        if Ensemble:
            set_seed(t)
            if dataset_name == "fed_isic2019":
                randomlist = [i for i in range(5) if i != t]
            else:
                randomlist = random.sample(range(0, 10), 5)
            print("ensemble", t, randomlist)
            for i in randomlist:
                model = train(strategy, mode="Ensemble/" + mode, num=str(i), pretrain=True)
                ensembles.append(model)
        else:
            # if os.path.exists(SAVE_PATH):
                # load_model(SAVE_PATH, algclass)
            print("trail", t)
            algclass = train(strategy, mode="Ensemble/"+ mode , num=str(t), pretrain=True)
            # else:
            #     print("ERROR:model not found")

        train_loaders = train_dataset()
        test_loaders = local_test_datasets() + global_test_dataset()

        n_data, accs, sp_ents, sp_vars, auroc_ents, auroc_vars = [], [], [], [], [], []
        sp_ent_arrays = []
        res_list = [accs, sp_ents, sp_vars, auroc_ents, auroc_vars]
        res_names = "accs, sp_ents, sp_vars, auroc_ents, auroc_vars".split(", ")
        for client_idx in range(NUM_CLIENTS):
            n_data.append(test_loaders[client_idx].dataset.__len__())
            if Ensemble and strategy == "pooled":
                model = [algclass.client_model[0] for algclass in ensembles]
            elif Ensemble:
                model = [algclass.client_model[client_idx] for algclass in ensembles]
            elif strategy == "pooled":
                model = algclass.client_model[0]
            else:
                model = algclass.client_model[client_idx]
            acc, sp_ent, sp_var, auroc_ent, auroc_var, ent = fine_tune_and_test(model, train_loaders[client_idx],
                                                                           [test_loaders[client_idx]], device, MCDO=MCDO,
                                                                           Ensemble=Ensemble, FT=FT,
                                                                           st=strategy)
            res = [acc, sp_ent, sp_var, auroc_ent, auroc_var]
            
            for i, val in enumerate(res):
                res_list[i].append(val)
            sp_ent_array = plot_selective_prediction_range(model, train_loaders[client_idx],
                                                            [test_loaders[client_idx]], device, MCDO=MCDO,
                                                            Ensemble=Ensemble, FT=FT)
            sp_ent_arrays.append(sp_ent_array)
        final_res = []
        for i, res in enumerate(res_list):
            if "var" in res_names[i] and not uncertainty:
                final_res.append(0)
                continue
            print(res_names[i])
            print(res)
            final_res.append(get_weighted_avg(res, n_data, eta=eta))
            print("weighted average", get_weighted_avg(res, n_data))
            print("eta weighted average", get_weighted_avg(res, n_data, eta=eta))
            
        arr = []
        for i in range(len(sp_ent_arrays[0])):
            arr.append(get_weighted_avg([sp_ent_arrays[c][i] for c in range(NUM_CLIENTS)], n_data, eta=eta))
        return res_names, final_res, arr
        
            
    res_5_run, arr_5_run = [], []
    for t in range(5):
        res_names, final_res, arr = fine_tune_test_one_run(t)
        res_5_run.append(final_res)
        arr_5_run.append(arr)
    avg_across_5_run = np.mean(res_5_run, axis=0)
    std_across_5_run = np.std(res_5_run, axis=0)
    print("_".join([strategy+f, method]), "=", list(np.mean(arr_5_run, axis=0)))
    print("with std_dev", list(np.std(arr_5_run, axis=0)))

    if save:
        # saving to json file
        for i, res_name in enumerate(res_names):
            if "var" in res_name and not uncertainty:
                continue
            print(res_name)
            res_report = ('%.3f' % avg_across_5_run[i] ) + " += " + ('%.3f' % std_across_5_run[i])
            print(res_report)
            res_dict["_".join([strategy+f, method, res_name])] = res_report
        with open(file_name, "wt") as wf:
            wf.write(json.dumps(res_dict))
            print(res_dict)


def analyze_distribution(train_test="train"):
    if train_test == "train": datasets = train_dataset()
    else: datasets = local_test_datasets()
    # plt.figure(figsize=(15, 10))
    print([data_loader.dataset.__len__() for data_loader in datasets])
    c = []
    for i, data_loader in enumerate(datasets):
        y_true = []
        for (X, y) in data_loader:
            y_true.append(y)
        y_true_final = np.concatenate(y_true)
        count = collections.Counter(list(y_true_final.flatten()))
        # ax = plt.subplot(3, 2, i+1)
        # ax.bar(range(2), [count[i] for i in range(2)])
        # ax.set_title("client {}".format(i))
        print("client_{}_{} =".format(i, train_test), [count[v] for v in range(8)])
        if train_test == "test":
            print("data_{} = [client_{}_train ; client_{}_test]' ".format(i, i, i))
        c.append(len(y_true_final))
    # plt.savefig("figs/audio_{}_data.png".format(train_test))
    # print("total number of samples:" + train_test, sum(c), c)


def centralised_5runtest(strategy, device="cuda" if torch.cuda.is_available() else "cpu", MCDO=False,
                         Ensemble=False, mode=""):
    method = "base"
    if MCDO: method = "Dropout"
    elif Ensemble: method = "Ensemble"
    elif earlyexit: method = "earlyexit"

    if os.path.exists(dataset_name + "_centralised.json"):
        res_dict = json.loads(open(dataset_name + "_centralised.json").readline())
        print(res_dict)
    else:
        res_dict = {}
        
    if "_".join([strategy, method, "auroc_ents"]) in res_dict:
        print("strategy {}, {} already run".format(strategy, method))
        return
    

    res_5_run = []
    for t in range(5):
        if Ensemble:
            res_names, final_res = ensemble(strategy, device, seed=t)
        else:
            res_names, final_res = MCD_evaluation(strategy, device, MCDO, num=str(t), mode=mode)
        res_5_run.append(final_res)

    avg_across_5_run = np.mean(res_5_run, axis=0)
    std_across_5_run = np.std(res_5_run, axis=0)
    for i, res_name in enumerate(res_names):
        print(res_name)
        res_report = ('%.3f' % avg_across_5_run[i] ) + " += " + ('%.3f' % std_across_5_run[i])
        print(res_report)
        res_dict["_".join([strategy, method, res_name])] = res_report
    with open(dataset_name + "_centralised.json", "wt") as wf:
        wf.write(json.dumps(res_dict))
        print(res_dict)

    
def report_values(centralised="", one_table=False, std=True, compare_uq=False, average=False):
    if average:
        filename = dataset_name + centralised + "_average.json"
    else:
        filename = dataset_name + centralised + ".json"
    if os.path.exists(filename):
        res_dict = json.loads(open(filename).readline())
        print(res_dict)
    else:
        print("result not exist")
        return
    if len(centralised) > 1:
        strategies = ["pooled",  "fedavg"]
    else:
        strategies = ["fedavg", "fedprox", "fedavg-FT", "fedprox-FT", "fedbn",  "fedbn-FT", "fedap", "fedap-FT"]
        # strategies = ["fedavg", "fedprox", "fedavg-FT", "fedprox-FT"]

    
    if std is False:
        for k in res_dict:
            res_dict[k] = res_dict[k].split(" += ")[0]
    if one_table:
        for strategy in strategies:
            print(strategy, " & ".join([res_dict["_".join([strategy, method, "auroc_ents"])] for method in ["base", "Dropout", "Ensemble"]]), end=" & ")
            print(" & ".join([res_dict["_".join([strategy, method, "sp_ents"])] for method in ["base", "Dropout", "Ensemble"]]))
    elif compare_uq:
        for strategy in strategies:
            print(strategy)
            for method in ["base", "Dropout", "Ensemble", "fedee"][:]:
                print(method, " & ".join([res_dict["_".join([strategy, method, met])] for met in ["accs", "auroc_ents", "sp_ents"]]))
    else:
        for met in ["accs", "auroc_ents", "sp_ents"]:
            print(met)
            for strategy in strategies:
                # if dataset_name == "fed_isic2019":
                #     print(strategy, " & ".join([res_dict["_".join([strategy, method, met])] for method in ["base", "Dropout"]]))
                # # elif dataset_name == "fed_har" and strategy in ["pooled", "pooled-FT", "fedavg", "fedavg-FT"]:
                # #     print(strategy, " & ".join([res_dict["_".join([strategy, method, met])] for method in ["base", "Dropout", "Ensemble", "earlyexit"]]))
                # else:
                print(strategy, " & ".join([res_dict["_".join([strategy, method, met])] for method in ["base", "Dropout", "Ensemble"]]))


def plot_selective_prediction_range(model, train_dataloader, test_dataloaders, device, MCDO=False, Ensemble=False, FT=0):
    from uncertainty import evaluate_model_on_tests, evaluate, selected_predict
    lossfunc = BaselineLoss()
    from util.traineval import train
    set_seed(0)
    if Ensemble:
        for m in model:
            optimizer = Optimizer(m.parameters(), lr=LR)
            for epoch in range(0, FT):
                train(m, train_dataloader, optimizer, lossfunc, device)
    else:
        optimizer = Optimizer(model.parameters(), lr=LR)  # 0.0001 for pooled?
        for epoch in range(0, FT):
            train(model, train_dataloader, optimizer, lossfunc, device)
        # train_prox(prox_arg(1e-3), model, global_model, train_dataloader, optimizer, lossfunc, device)
    dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(model, test_dataloaders, metric,
                                                                             return_pred=True, MCDO=MCDO,
                                                                             Ensemble=Ensemble,earlyexit=earlyexit,
                                                                             task=TASK)
    sp_ent = [selected_predict(entropy, y_true, y_pred, metric, percent) for percent in np.arange(0.0, 0.85, 0.05)]
    return sp_ent


def centralised_selective_prediction_range(strategy, device="cuda" if torch.cuda.is_available() else "cpu", mode="", MCDO=False,
                                           Ensemble=False):
    method = "base_"
    if MCDO: method = "Dropout_"
    elif Ensemble: method = "Ensemble_"
    elif earlyexit: method = "earlyexit"
    arr = []
    for t in range(5):
        ensembles = []
        if Ensemble:
            set_seed(t)
            randomlist = random.sample(range(0, 10), 5)
            print("ensemble", t, randomlist)
            for i in randomlist:
                model = train(strategy, mode="Ensemble/"+mode, num=str(i), pretrain=True)
                ensembles.append(model)
        else:
            # if os.path.exists(SAVE_PATH):
            # load_model(SAVE_PATH, algclass)
            print("trail", t)
            algclass = train(strategy, mode="Ensemble/"+mode, num=str(t), pretrain=True)

        train_loaders = get_pooled_train_loader()
        test_loaders = global_test_dataset()

        if Ensemble and strategy == "pooled":
            model = [algclass.client_model[0] for algclass in ensembles]
        elif Ensemble:
            model = [algclass.server_model for algclass in ensembles]
        elif strategy == "pooled":
            model = algclass.client_model[0]
        else:
            model = algclass.server_model
        sp_ent_array = plot_selective_prediction_range(model, train_loaders,
                                                       test_loaders, device, MCDO=MCDO,
                                                       Ensemble=Ensemble)
        arr.append(sp_ent_array)
    print(method + strategy, "=", list(np.mean(arr, axis=0)))

    
def base_cross_client():
    matrix = [[] for i in range(NUM_CLIENTS)]
    algclass = train("base", pretrain=True)
    # for model in algclass.client_models:
    test_dataloaders = local_test_datasets()
    for client_idx in range(NUM_CLIENTS):
        model = algclass.client_model[client_idx]
        for data_idx in range(NUM_CLIENTS):
            dict_cindex = evaluate_model_on_tests(model, [test_dataloaders[data_idx]],
                                                                                     metric, return_pred=False)
            matrix[client_idx].append(dict_cindex["client_test_0"])
    print(np.array(matrix))


def pcg_testing(mode):
    for alg in "base | pooled | fedavg | fedbn | fedprox | fedap".split(" | ")[:1]:
        train(alg, "cuda" if torch.cuda.is_available() else "cpu", mode=mode)
        # centralised_selective_prediction_range(alg)
        # centralised_selective_prediction_range(alg, MCDO=True)
        # centralised_selective_prediction_range(alg, Ensemble=True)
        # train(alg, "cuda" if torch.cuda.is_available() else "cpu", mode="har_clientsplit/50split/earlyexit/")
        # print("earlyexit")
        # MCD_evaluation(alg, "cuda" if torch.cuda.is_available() else "cpu", False, mode="har_clientsplit/50split/earlyexit/")
        # print("without earlyexit")
        # MCD_evaluation(alg, "cuda" if torch.cuda.is_available() else "cpu", False, mode="har_clientsplit/50split/")
        fine_tune_testing(alg, MCDO=False, Ensemble=False, FT=0,  mode=mode) #, mode="audio_normed/50split/earlyexit/")
        fine_tune_testing(alg, MCDO=False, Ensemble=False, FT=5,  mode=mode) #, mode="audio_normed/50split/earlyexit/")
        # train(alg, "cuda" if torch.cuda.is_available() else "cpu", mode="audio_new/")
        print("="*48)
        # fine_tune_testing(alg, MCDO=False, Ensemble=False, FT=0, mode="audio_normed/3layer/")


def earlyexit_testing(alg="fedavg", mode="earlyexit2D-yita/320/"):
    train(alg, "cuda" if torch.cuda.is_available() else "cpu", mode=mode)
    fine_tune_testing(alg, MCDO=False, Ensemble=False, FT=0,  mode=mode) #, mode="audio_normed/50split/earlyexit/")
    fine_tune_testing(alg, MCDO=False, Ensemble=False, FT=5,  mode=mode) #, mode="audio_normed/50split/earlyexit/")
    # centralised_selective_prediction_range(alg)
    # centralised_selective_prediction_range(alg, MCDO=True)
    # centralised_selective_prediction_range(alg, Ensemble=True)
    # train(alg, "cuda" if torch.cuda.is_available() else "cpu", mode="har_clientsplit/50split/earlyexit/")
    print("earlyexit")
    # MCD_evaluation(alg, "cuda" if torch.cuda.is_available() else "cpu", False, mode="har_clientsplit/50split/earlyexit/")
    # print("without earlyexit")
    # MCD_evaluation(alg, "cuda" if torch.cuda.is_available() else "cpu", False, mode="har_clientsplit/50split/")
    # train(alg, "cuda" if torch.cuda.is_available() else "cpu", mode="audio_new/")
    # fine_tune_testing(alg, MCDO=False, Ensemble=False, FT=0, mode="audio_normed/3layer/")
    print("="*48)


def formal_five_times(alg, mode, earlyexit=earlyexit, save=True, eta=np.e):
    if earlyexit: mode += "earlyexit/"
    # n_models = 5 if earlyexit else 10
    # for i in range(n_models):
    #     train(alg, mode="Ensemble/" + mode, num=str(i))
    fine_tune_testing_5run(alg, FT=0, mode=mode, save=save, eta=eta)
    if not earlyexit:
        fine_tune_testing_5run(alg, FT=0, mode=mode, MCDO=True, save=save, eta=eta)
        fine_tune_testing_5run(alg, FT=0, mode=mode, Ensemble=True, save=save, eta=eta)
    if alg != "base":
        fine_tune_testing_5run(alg, FT=5, mode=mode, save=save, eta=eta)
        if not earlyexit:
            fine_tune_testing_5run(alg, FT=5, mode=mode, MCDO=True, save=save, eta=eta)
            fine_tune_testing_5run(alg, FT=5, mode=mode, Ensemble=True, save=save, eta=eta)


def count_FLOP():
    from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
    from uncertainty import enable_dropout
    models = [Baseline(), Baseline(), BaselineEarlyExit()]
    methods = ["backbone", "MCDrop", "EarlyExit"]
    enable_dropout(models[1])
    test_dataloader_iterator = iter(global_test_dataset()[0])
    for (X, y) in test_dataloader_iterator:
        if torch.cuda.is_available():
            X = X.cuda()
        for i, model in enumerate(models):
            print("=" * 20, methods[i], "=" * 20)
            if torch.cuda.is_available(): model.to("cuda")
            flops = FlopCountAnalysis(model, X)
            # print("total flops", flops.total())
            # print(flops.by_operator())
            print(flop_count_table(flops))
            # print(flop_count_str(flops))
        break


def ood_testing(strategy, mode="", earlyexit=earlyexit):
    from uncertainty import get_uncertainty_on_tests, cal_AUROC
    from har_config import get_dataloader_oppo
    if earlyexit: mode += "earlyexit/"
    algclass = train(strategy, mode="Ensemble/"+ mode , num="0", pretrain=True)

    # model = algclass.client_model[0]
    # in_distribution = [local_test_datasets()[0]]
    # local_ood = local_test_datasets()[1:]
    # global_ood = [get_dataloader_oppo()]
    # for test_loader in [in_distribution, local_ood, global_ood]:
    #     var, ent = get_uncertainty_on_tests(model, test_loader, return_pred=True, earlyexit=earlyexit, task=TASK)
    #     val = []
    #     for k in var:
    #         # print(np.mean(var[k]))
    #         # val += list(var[k])
    #         val += list(ent[k])
    #     # print(val)
    res = []
    test_loaders = local_test_datasets()
    for num in range(5):
        n_data, client_res = [], []
        algclass = train(strategy, mode="Ensemble/"+ mode , num=str(num), pretrain=True)
        
        # global_model = algclass.server_model
        for client_idx in range(NUM_CLIENTS):
            local_model = algclass.client_model[client_idx]
            n_data.append(test_loaders[client_idx].dataset.__len__())
            _, ent_in = get_uncertainty_on_tests(local_model, [test_loaders[client_idx]], return_pred=True, earlyexit=earlyexit, task=TASK)
            _, ent_out = get_uncertainty_on_tests(local_model, [get_dataloader_oppo()], return_pred=True, earlyexit=earlyexit, task=TASK)
            auroc = cal_AUROC(list(ent_in["client_test_0"]) + list(ent_out["client_test_0"]), range(len(list(ent_in["client_test_0"]))))
            client_res.append(auroc)
        res.append(get_weighted_avg(client_res, n_data, eta=1))
    print(np.mean(res), np.std(res))



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default="fedavg",
                        help='Algorithm to choose: [base | fedavg | fedbn | fedprox | fedap | metafed | pooled ]')
    # # # parser.add_argument('--mu', type=int, default=1)
    parser.add_argument('--num', type=str, default="0")
    args = parser.parse_args()
    alg = args.alg
    num = args.num

    # analyze_distribution()
    # analyze_distribution("test")
    train(alg, mode="Ensemble/", num="0")
    
    for alg in "base | pooled | fedavg | fedprox | fedbn | fedap".split(" | "):
        formal_five_times(alg, mode="", earlyexit=earlyexit)
    report_values(compare_uq=True)
    count_FLOP()
