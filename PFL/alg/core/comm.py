# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import copy

def communication(args, server_model, models, client_weights):
    client_num=len(models)
    with torch.no_grad():
        if args.alg.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key and 'exit' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.alg.lower()=='fedee':
            for key in server_model.state_dict().keys():
                # print(key)
                if 'exit' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.alg.lower()=='fedap':
            tmpmodels=[]
            for i in range(client_num):
                tmpmodels.append(copy.deepcopy(models[i]).to(args.device))
            with torch.no_grad():
                for cl in range(client_num):
                    for key in server_model.state_dict().keys():
                        temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                        for client_idx in range(client_num):
                            temp += client_weights[cl,client_idx] * tmpmodels[client_idx].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)
                        if 'bn' not in key and 'exit' not in key:
                            models[cl].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.alg.lower()=='fedapee':
            tmpmodels=[]
            for i in range(client_num):
                tmpmodels.append(copy.deepcopy(models[i]).to(args.device))
            with torch.no_grad():
                for cl in range(client_num):
                    for key in server_model.state_dict().keys():
                        temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                        for client_idx in range(client_num):
                            temp += client_weights[cl,client_idx] * tmpmodels[client_idx].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)
                        if 'bn' not in key and 'exit' not in key:
                            models[cl].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.alg.lower()=='fedeeap':
            # fedee + similarity weights, do not keep BN locally
            tmpmodels=[]
            for i in range(client_num):
                tmpmodels.append(copy.deepcopy(models[i]).to(args.device))
            with torch.no_grad():
                for cl in range(client_num):
                    for key in server_model.state_dict().keys():
                        temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                        for client_idx in range(client_num):
                            temp += client_weights[cl,client_idx] * tmpmodels[client_idx].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)
                        if 'exit' not in key:
                            models[cl].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # print("server", key, server_model.state_dict()[key].get_device())
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    # print("temp device:", temp.get_device())
                    for client_idx in range(len(client_weights)):
                        # print("client_" + str(client_idx), key, models[client_idx].state_dict()[key].get_device())
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        if 'exit' not in key:
                            models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models
    