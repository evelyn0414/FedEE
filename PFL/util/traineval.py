import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from datautil.datasplit import define_pretrain_dataset
# from datautil.prepare_data import get_whole_dataset
# from datautil.importdata import pretrain_dataset
from flamby.utils import evaluate_model_on_tests

def train(model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    # total = 0
    # correct = 0
    for data, target in data_loader:
        data = data.to(device).float()
        # target = target.to(device).long()
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        # total += target.size(0)
        # pred = output.data.max(1)[1]
        # correct += pred.eq(target.view(-1)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # print("train loss", loss_all)
    return loss_all / len(data_loader), None


def test(model, data_loader, loss_fun, device, metric=None):
    model.eval()
    loss_all = 0
    res = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device).float()
            target = target.to(device)
            # target = target.to(device).long()
            output = model(data)
            loss = loss_fun(output, target)
            loss_all += loss.item()
            # total += target.size(0)
            # pred = output.data.max(1)[1]
            # correct += pred.eq(target.view(-1)).sum().item()
        if metric:
            res = evaluate_model_on_tests(model, [data_loader], metric)
            res = res["client_test_0"]

        return loss_all / len(data_loader), res


def train_prox(args, model, server_model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for step, (data, target) in enumerate(data_loader):
        data = data.to(device).float()
        # target = target.to(device).long()
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        if step > 0:
            w_diff = torch.tensor(0., device=device)
            # print("model.name = ", model.name)
            if model.name == "earlyexit":
                for w, w_t in zip(server_model.base_model.parameters(), model.base_model.parameters()):
                    w_diff += torch.pow(torch.norm(w - w_t), 2)
                # print("fedprox edited")
            else:
                for w, w_t in zip(server_model.parameters(), model.parameters()):
                    w_diff += torch.pow(torch.norm(w - w_t), 2)

            w_diff = torch.sqrt(w_diff)
            loss += args.mu / 2. * w_diff

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        # total += target.size(0)
        # pred = output.data.max(1)[1]
        # correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), 0


def trainwithteacher(model, data_loader, optimizer, loss_fun, device, tmodel, lam, args, flag):
    model.train()
    if tmodel:
        tmodel.eval()
        if not flag:
            with torch.no_grad():
                for key in tmodel.state_dict().keys():
                    if 'num_batches_tracked' in key:
                        pass
                    elif args.nosharebn and 'bn' in key:
                        pass
                    else:
                        model.state_dict()[key].data.copy_(
                            tmodel.state_dict()[key])
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device).float()
        # target = target.to(device).long()
        target = target.to(device)
        output = model(data)
        f1 = model.get_sel_fea(data, args.plan)
        loss = loss_fun(output, target)
        if flag and tmodel:
            f2 = tmodel.get_sel_fea(data, args.plan).detach()
            loss += (lam*F.mse_loss(f1, f2))
        loss_all += loss.item()
        # total += target.size(0)
        # pred = output.data.max(1)[1]
        # correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), 0


# def pretrain_model(args, model, filename, device='cuda', train_loaders=None):
#     print('===training pretrained model===')
#     # if train_loaders is None:
#     data = get_whole_dataset(args.dataset)(args)
#     predata = define_pretrain_dataset(args, data)
#     traindata = torch.utils.data.DataLoader(
#         predata, batch_size=args.batch, shuffle=True)
#     # else:
#     #     traindata = pretrain_dataset(args)
#     loss_fun = nn.CrossEntropyLoss()
#     opt = optim.SGD(params=model.parameters(), lr=args.lr)
#     for _ in range(args.pretrained_iters):
#         _, acc = train(model, traindata, opt, loss_fun, device)
#     torch.save({
#         'state': model.state_dict(),
#         'acc': acc
#     }, filename)
#     print('===done!===')
