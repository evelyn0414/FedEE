import os
import copy
import math
import numpy as np
import torch

from alg.fedavg import fedavg
from alg.fedbn import fedbn
# from util.traineval import pretrain_model
import torch.nn as nn
import torch.optim as optim

class fedapee(fedavg):
    def __init__(self, args, model=None, loss=nn.CrossEntropyLoss(), optimizer=optim.SGD):
        super(fedapee, self).__init__(args, model, loss, optimizer)
        self.pretrain_model = copy.deepcopy(model).to(self.args.device)

    def set_client_weight(self, train_loaders):
        os.makedirs('./checkpoint/'+'pretrained/', exist_ok=True)
        preckpt = './checkpoint/'+'pretrained/' + \
            self.args.dataset +str(self.args.batch)
            # self.args.dataset+'_'+str(self.args.batch)
        # self.pretrain_model = copy.deepcopy(
        #     self.model).to(self.args.device)
        
        if self.args.dataset != "fed_isic2019":
            self.args.alg = "fedbn"
            bn = fedbn(self.args, self.pretrain_model, self.loss_fun)
            for _ in range(2):
                for wi in range(20):
                    for client_idx in range(self.args.n_clients):
                        bn.client_train(client_idx, train_loaders[client_idx], 0)
            bn.server_aggre()
            self.pretrain_model = bn.server_model
            self.args.alg = "fedapee"
        
        torch.save({
            'state': self.pretrain_model.state_dict(),
            # 'acc': acc
        }, preckpt)

        # if not os.path.exists(preckpt + "umm"):
        #     pretrain_model(self.args, self.pretrain_model,
        #                    preckpt, self.args.device, train_loaders)

        self.preckpt = preckpt
        self.client_weight = get_weight_preckpt(
            self.args, self.pretrain_model, self.preckpt, train_loaders, self.client_weight, device=self.args.device)
        print(self.client_weight)


def get_form(model):
    tmpm = []
    tmpv = []
    for name in model.state_dict().keys():
        if 'running_mean' in name:
            # print("tmpm", name)
            tmpm.append(model.state_dict()[name].detach().to('cpu').numpy())
        if 'running_var' in name:
            # print("tmpv", name)
            tmpv.append(model.state_dict()[name].detach().to('cpu').numpy())
    return tmpm, tmpv


def get_wasserstein(m1, v1, m2, v2, mode='nosquare'):
    w = 0
    bl = len(m1)
    for i in range(bl):
        tw = 0
        tw += (np.sum(np.square(m1[i]-m2[i])))
        tw += (np.sum(np.square(np.sqrt(v1[i]) - np.sqrt(v2[i]))))
        if mode == 'square':
            w += tw
        else:
            w += math.sqrt(tw)
    return w


def get_weight_matrix1(args, bnmlist, bnvlist, client_weights):
    client_num = len(bnmlist)
    weight_m = np.zeros((client_num, client_num))
    for i in range(client_num):
        for j in range(client_num):
            if i == j:
                weight_m[i, j] = 0
            else:
                tmp = get_wasserstein(
                    bnmlist[i], bnvlist[i], bnmlist[j], bnvlist[j])
                if tmp == 0:
                    weight_m[i, j] = 100000000000000
                else:
                    weight_m[i, j] = 1/tmp
    weight_s = np.sum(weight_m, axis=1)
    weight_s = np.repeat(weight_s, client_num).reshape(
        (client_num, client_num))
    weight_m = (weight_m/weight_s)*(1-args.model_momentum)
    for i in range(client_num):
        weight_m[i, i] = args.model_momentum
    return weight_m


def get_weight_preckpt(args, model, preckpt, trainloadrs, client_weights, device='cuda'):
    model.load_state_dict(torch.load(preckpt)['state'])
    model.eval()
    bnmlist1, bnvlist1 = [], []
    for i in range(args.n_clients):
        avgmeta = metacount(get_form(model)[0])
        with torch.no_grad():
            for data, _ in trainloadrs[i]:
                data = data.to(device).float()
                fea = model.getallfea(data)
                nl = len(data)
                tm, tv = [], []
                for item in fea:
                    # print("item.shape =", item.shape)
                    if len(item.shape) == 5:
                        # print("shape = 5 yes")
                        tm.append(torch.mean(
                            item, dim=[0, 2, 3, 4]).detach().to('cpu').numpy())
                        tv.append(
                            torch.var(item, dim=[0, 2, 3, 4]).detach().to('cpu').numpy())
                    elif len(item.shape) == 4:
                        tm.append(torch.mean(
                            item, dim=[0, 2, 3]).detach().to('cpu').numpy())
                        tv.append(
                            torch.var(item, dim=[0, 2, 3]).detach().to('cpu').numpy())
                    elif len(item.shape) == 3:
                        tm.append(torch.mean(
                            item, dim=[0, 2]).detach().to('cpu').numpy())
                        tv.append(
                            torch.var(item, dim=[0, 2]).detach().to('cpu').numpy())
                    else:
                        tm.append(torch.mean(
                            item, dim=0).detach().to('cpu').numpy())
                        tv.append(
                            torch.var(item, dim=0).detach().to('cpu').numpy())
                # print("tm =", tm)
                # print("tv =", tv)
                # print("nl =", nl)
                avgmeta.update(nl, tm, tv)
        bnmlist1.append(avgmeta.getmean())
        bnvlist1.append(avgmeta.getvar())
    weight_m = get_weight_matrix1(args, bnmlist1, bnvlist1, client_weights)
    return weight_m


class metacount(object):
    def __init__(self, numpyform):
        super(metacount, self).__init__()
        self.count = 0
        self.mean = []
        self.var = []
        self.bl = len(numpyform)
        for i in range(self.bl):
            self.mean.append(np.zeros(len(numpyform[i])))
            self.var.append(np.zeros(len(numpyform[i])))

    def update(self, m, tm, tv):
        tmpcount = self.count+m
        for i in range(self.bl):
            # print(i)
            # print("self.mean[i].shape =", self.mean[i].shape)
            # print("self.count =", self.count)
            # print("tm[i].shape =", tm[i].shape)
            # print("m =", m)
            # print("tmpcount =", tmpcount)
            tmpm = (self.mean[i]*self.count + tm[i]*m)/tmpcount
            self.var[i] = (self.count*(self.var[i]+np.square(tmpm -
                           self.mean[i])) + m*(tv[i]+np.square(tmpm-tm[i])))/tmpcount
            self.mean[i] = tmpm
        self.count = tmpcount

    def getmean(self):
        return self.mean

    def getvar(self):
        return self.var


class fedeeap(fedavg):
    def __init__(self, args, model=None, loss=nn.CrossEntropyLoss(), optimizer=optim.SGD):
        super(fedeeap, self).__init__(args, model, loss, optimizer)

    def set_client_weight(self, train_loaders):
        os.makedirs('./checkpoint/'+'pretrained/', exist_ok=True)
        preckpt = './checkpoint/'+'pretrained/' + \
            self.args.dataset +str(self.args.batch)
            # self.args.dataset+'_'+str(self.args.batch)
        self.pretrain_model = copy.deepcopy(
            self.server_model).to(self.args.device)
        
        if self.args.dataset != "fed_isic2019":
            self.args.alg = "fedbn"
            bn = fedbn(self.args, self.pretrain_model, self.loss_fun)
            for _ in range(2):
                for wi in range(20):
                    for client_idx in range(self.args.n_clients):
                        bn.client_train(client_idx, train_loaders[client_idx], 0)
            bn.server_aggre()
            self.pretrain_model = bn.server_model
            self.args.alg = "fedeeap"
        
        torch.save({
            'state': self.pretrain_model.state_dict(),
            # 'acc': acc
        }, preckpt)

        # if not os.path.exists(preckpt + "umm"):
        #     pretrain_model(self.args, self.pretrain_model,
        #                    preckpt, self.args.device, train_loaders)

        self.preckpt = preckpt
        self.client_weight = get_weight_preckpt(
            self.args, self.pretrain_model, self.preckpt, train_loaders, self.client_weight, device=self.args.device)
        print(self.client_weight)


def get_form(model):
    tmpm = []
    tmpv = []
    for name in model.state_dict().keys():
        if 'running_mean' in name:
            # print("tmpm", name)
            tmpm.append(model.state_dict()[name].detach().to('cpu').numpy())
        if 'running_var' in name:
            # print("tmpv", name)
            tmpv.append(model.state_dict()[name].detach().to('cpu').numpy())
    return tmpm, tmpv


def get_wasserstein(m1, v1, m2, v2, mode='nosquare'):
    w = 0
    bl = len(m1)
    for i in range(bl):
        tw = 0
        tw += (np.sum(np.square(m1[i]-m2[i])))
        tw += (np.sum(np.square(np.sqrt(v1[i]) - np.sqrt(v2[i]))))
        if mode == 'square':
            w += tw
        else:
            w += math.sqrt(tw)
    return w


def get_weight_matrix1(args, bnmlist, bnvlist, client_weights):
    client_num = len(bnmlist)
    weight_m = np.zeros((client_num, client_num))
    for i in range(client_num):
        for j in range(client_num):
            if i == j:
                weight_m[i, j] = 0
            else:
                tmp = get_wasserstein(
                    bnmlist[i], bnvlist[i], bnmlist[j], bnvlist[j])
                if tmp == 0:
                    weight_m[i, j] = 100000000000000
                else:
                    weight_m[i, j] = 1/tmp
    weight_s = np.sum(weight_m, axis=1)
    weight_s = np.repeat(weight_s, client_num).reshape(
        (client_num, client_num))
    weight_m = (weight_m/weight_s)*(1-args.model_momentum)
    for i in range(client_num):
        weight_m[i, i] = args.model_momentum
    return weight_m


def get_weight_preckpt(args, model, preckpt, trainloadrs, client_weights, device='cuda'):
    model.load_state_dict(torch.load(preckpt)['state'])
    model.eval()
    bnmlist1, bnvlist1 = [], []
    for i in range(args.n_clients):
        avgmeta = metacount(get_form(model)[0])
        with torch.no_grad():
            for data, _ in trainloadrs[i]:
                data = data.to(device).float()
                fea = model.getallfea(data)
                nl = len(data)
                tm, tv = [], []
                for item in fea:
                    # print("item.shape =", item.shape)
                    if len(item.shape) == 5:
                        # print("shape = 5 yes")
                        tm.append(torch.mean(
                            item, dim=[0, 2, 3, 4]).detach().to('cpu').numpy())
                        tv.append(
                            torch.var(item, dim=[0, 2, 3, 4]).detach().to('cpu').numpy())
                    elif len(item.shape) == 4:
                        tm.append(torch.mean(
                            item, dim=[0, 2, 3]).detach().to('cpu').numpy())
                        tv.append(
                            torch.var(item, dim=[0, 2, 3]).detach().to('cpu').numpy())
                    elif len(item.shape) == 3:
                        tm.append(torch.mean(
                            item, dim=[0, 2]).detach().to('cpu').numpy())
                        tv.append(
                            torch.var(item, dim=[0, 2]).detach().to('cpu').numpy())
                    else:
                        tm.append(torch.mean(
                            item, dim=0).detach().to('cpu').numpy())
                        tv.append(
                            torch.var(item, dim=0).detach().to('cpu').numpy())
                # print("tm =", tm)
                # print("tv =", tv)
                # print("nl =", nl)
                avgmeta.update(nl, tm, tv)
        bnmlist1.append(avgmeta.getmean())
        bnvlist1.append(avgmeta.getvar())
    weight_m = get_weight_matrix1(args, bnmlist1, bnvlist1, client_weights)
    return weight_m


class metacount(object):
    def __init__(self, numpyform):
        super(metacount, self).__init__()
        self.count = 0
        self.mean = []
        self.var = []
        self.bl = len(numpyform)
        for i in range(self.bl):
            self.mean.append(np.zeros(len(numpyform[i])))
            self.var.append(np.zeros(len(numpyform[i])))

    def update(self, m, tm, tv):
        tmpcount = self.count+m
        for i in range(self.bl):
            # print(i)
            # print("self.mean[i].shape =", self.mean[i].shape)
            # print("self.count =", self.count)
            # print("tm[i].shape =", tm[i].shape)
            # print("m =", m)
            # print("tmpcount =", tmpcount)
            tmpm = (self.mean[i]*self.count + tm[i]*m)/tmpcount
            self.var[i] = (self.count*(self.var[i]+np.square(tmpm -
                           self.mean[i])) + m*(tv[i]+np.square(tmpm-tm[i])))/tmpcount
            self.mean[i] = tmpm
        self.count = tmpcount

    def getmean(self):
        return self.mean

    def getvar(self):
        return self.var
