import torch
import torch.nn as nn
import torch.nn.functional as func
from collections import OrderedDict, Counter
import numpy as np
from sklearn import metrics
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from sklearn.model_selection import train_test_split
import torchmetrics.functional as tm
from earlyexit import ExitBlock

NUM_CLIENTS = 8
BATCH_SIZE = 32
NUM_EPOCHS_POOLED = 20
NUM_EPOCHS_CENTRALIZED = 40
LR = 0.0003
Optimizer = torch.optim.Adam
SKIP_PERCENT = 0.0

pp_dict = {1: "lying", 2: "sitting", 3: "standing", 4: "walking", 12: "ascending stairs", 13: "descending stairs", 16: "vacuum cleaning", 17: "ironing"}

label_dict = {'ironing': 0, 'lying': 1, 'sitting': 2, 'standing': 3, 'walking': 4, 'walk':4, 'ascending stairs': 5, 'climb_stairs': 5,'descending stairs': 6, 'vacuum cleaning': 7, "getup_bed":8, "pour_water":9, "drink_glass":10}


class BaselineLoss(_Loss):
    """Weighted focal loss
    See this [link](https://amaarora.github.io/2020/06/29/FocalLoss.html) for
    a good explanation
    Attributes
    ----------
    alpha: torch.tensor of size 8, class weights
    gamma: torch.tensor of size 1, positive float, for gamma = 0 focal loss is
    the same as CE loss, increases gamma reduces the loss for the "hard to classify
    examples"
    """

    def __init__(
            self,
            alpha=torch.tensor(
                [1,1,1,1,1,1,1,1]
            ),
            gamma=2.0,
    ):
        super(BaselineLoss, self).__init__()
        self.alpha = alpha.to(torch.float)
        self.gamma = gamma

    def forward(self, inputs, targets):
        """Weighted focal loss function
        Parameters
        ----------
        inputs : torch.tensor of size 8, logits output by the model (pre-softmax)
        targets : torch.tensor of size 1, int between 0 and 7, groundtruth class
        """
        targets = targets.view(-1, 1).type_as(inputs)
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.long())
        logpt = logpt.view(-1)
        pt = logpt.exp()
        self.alpha = self.alpha.to(targets.device)
        at = self.alpha.gather(0, targets.data.view(-1).long())
        logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt

        return loss.mean()


def metric(y_true, logits):
    y_true = y_true.reshape(-1)
    # print(y_true.shape, logits.shape)
    preds = np.argmax(logits, axis=1)
    # print(y_true.shape, preds.shape)
    # return metrics.balanced_accuracy_score(y_true, preds)
    return metrics.accuracy_score(y_true, preds)
    # return metrics.f1_score(y_true, preds, average='macro')




def apply_label_map(y, label_map):
    """
    Apply a dictionary mapping to an array of labels
    Can be used to convert str labels to int labels
    Parameters:
        y
            1D array of labels
        label_map
            a label dictionary of (label_original -> label_new)
    Return:
        y_mapped
            1D array of mapped labels
            None values are present if there is no entry in the dictionary
    """

    y_mapped = []
    for l in y:
        y_mapped.append(label_map.get(l))
    return np.array(y_mapped)


def tmp_har():
    data_dir = "datautil/HAR/"
    datasets = ["pamap", "adl", "oppo", "realworld", "wisdm"]
    for dataset in datasets[:1]:
        print("")
        print(dataset)
        X_feats = np.load(data_dir + dataset + "/NoSSL_feats.npy")
        X = np.load(data_dir + dataset + "/X.npy")
        Y = np.load(data_dir + dataset + "/Y.npy")
        pid = np.load(data_dir + dataset + "/pid.npy")
        # print("raw input shape", X.shape)
        # print("feature shape", X_feats.shape)
        classes = Counter(list(Y))
        pid_count = Counter(list(pid))
        # print("classes", dict(classes))
        # print("subjects", dict(pid_count))

        for p in pid_count:
            count = Counter()
            print(p)
            for i in range(X.shape[0]):
                if pid[i] == p:
                    count[Y[i]] += 1
            print(dict(count))
        break


def get_mean_std_from_user_list_format(train_data):
    means = np.mean(train_data, axis=0)
    stds = np.std(train_data, axis=0)
    return (means, stds)


def normalise(data, mean, std):
    """
    Normalise data (Z-normalisation)
    """
    return ((data - mean) / std)


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    return (1435 // NUM_CLIENTS // batch_size) * NUM_EPOCHS_CENTRALIZED // num_updates


def get_dataloader(center: int = 0,
                   train: bool = True,
                   pooled: bool = False):
    test_split_proportion = 0.5
    # data_dir = "datautil/HAR/"
    data_dir = "/home/yz798/rds/hpc-work/data_util/HAR/"
    datasets = ["pamap", "adl", "oppo", "realworld", "wisdm"]
    dataset = datasets[0]
    X = np.load(data_dir + dataset + "/X.npy")
    Y = np.load(data_dir + dataset + "/Y.npy")
    pid = np.load(data_dir + dataset + "/pid.npy")

    pids = [103, 101, 104, 108, 107, 106, 105, 102]

    if pooled:
        train_x, test_x, train_y, test_y = [], [], [], []
        for center in range(NUM_CLIENTS):
            X_client = np.array([x for i, x in enumerate(X) if pid[i] == pids[center]])
            Y_client = np.array([y for i, y in enumerate(Y) if pid[i] == pids[center]])
            train_x1, test_x1, train_y1, test_y1 = train_test_split(X_client, Y_client, test_size=test_split_proportion, random_state=42)
            train_x.append(train_x1)
            test_x.append(test_x1)
            train_y.append(train_y1)
            test_y.append(test_y1)
        train_x = np.concatenate(train_x)
        test_x = np.concatenate(test_x)
        train_y = np.concatenate(train_y)
        test_y = np.concatenate(test_y)
    else:
        X = np.array([x for i, x in enumerate(X) if pid[i] == pids[center]])
        Y = np.array([y for i, y in enumerate(Y) if pid[i] == pids[center]])
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=test_split_proportion, random_state=42)



    classes = Counter(list(train_y))
    # print("train dist", dict(classes))
    classes = Counter(list(test_y))
    # print("test dist", dict(classes))

    means, stds = get_mean_std_from_user_list_format(train_x)
    train_x_norm = normalise(train_x, means, stds)
    test_x_norm = normalise(test_x, means, stds)

    if train:
        X, Y = train_x_norm, train_y
    else:
        X, Y = test_x_norm, test_y

    # X = X[:, np.newaxis, :, :]
    X = np.swapaxes(X,1,2)

    Y = apply_label_map(Y, pp_dict)
    Y = apply_label_map(Y, label_dict)
    # print(X.shape, Y.shape)

    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(Y)

    data = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    return data_loader


def increase_dimension(A, target_dim=1000):
    from scipy.ndimage.interpolation import map_coordinates
    new_dims = []
    for original_length, new_length in zip(A.shape, (A.shape[0], A.shape[1], target_dim)):
        new_dims.append(np.linspace(0, original_length-1, new_length))

    coords = np.meshgrid(*new_dims, indexing='ij')
    B = map_coordinates(A, coords)
    # print(A.shape, np.mean(A[1]))
    # print(B.shape, np.mean(B[1]))
    return B


def get_dataloader_oppo():
    data_dir = "/home/yz798/rds/hpc-work/data_util/HAR/"
    datasets = ["pamap", "adl", "oppo", "realworld", "wisdm"]

    # for dataset in datasets:
    #     print(dataset)
    #     X = np.load(data_dir + dataset + "/X.npy")
    #     Y = np.load(data_dir + dataset + "/Y.npy")
    #     print(X.shape, Y.shape)
    # can try wisdm or adl
    dataset = datasets[2] 
    X = np.load(data_dir + dataset + "/X.npy")
    Y = np.load(data_dir + dataset + "/Y.npy")
    print(X.shape, Y.shape)
    dict_map = {}
    num = 0
    for y in Y:
        if y not in dict_map:
            dict_map[y] = num
            num += 1
    # print(dict_map)
    Y = [dict_map[y] for y in Y]
    # print(Y)
    # print(X.shape, Y.shape)
    # pid = np.load(data_dir + dataset + "/pid.npy")
    # print(pid)
    

    # classes = Counter(list(Y))
    # print("class dist", dict(classes))

    means, stds = get_mean_std_from_user_list_format(X)
    X = normalise(X, means, stds)

    X = np.swapaxes(X,1,2)
    X = increase_dimension(X)

    # Y = apply_label_map(Y, pp_dict)
    # Y = apply_label_map(Y, label_dict)
    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(Y)

    data = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    return data_loader
#
# class PamapModel(nn.Module):
#     def __init__(self, n_feature=64, out_dim=10):
#         super(PamapModel, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_channels=27, out_channels=16, kernel_size=(1, 9))
#         self.bn1 = nn.BatchNorm2d(16)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
#         self.conv2 = nn.Conv2d(
#             in_channels=16, out_channels=32, kernel_size=(1, 9))
#         self.bn2 = nn.BatchNorm2d(32)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
#         self.fc1 = nn.Linear(in_features=32*44, out_features=n_feature)
#         self.fc1_relu = nn.ReLU()
#         self.fc2 = nn.Linear(in_features=n_feature, out_features=out_dim)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool1(self.relu1(self.bn1(x)))
#         x = self.conv2(x)
#         x = self.pool2(self.relu2(self.bn2(x)))
#         x = x.reshape(-1, 32 * 44)
#         feature = self.fc1_relu(self.fc1(x))
#         out = self.fc2(feature)
#         return out
#
#     def getallfea(self, x):
#         fealist = []
#         x = self.conv1(x)
#         fealist.append(x.clone().detach())
#         x = self.pool1(self.relu1(self.bn1(x)))
#         x = self.conv2(x)
#         fealist.append(x.clone().detach())
#         return fealist
#
#     def getfinalfea(self, x):
#         x = self.conv1(x)
#         x = self.pool1(self.relu1(self.bn1(x)))
#         x = self.conv2(x)
#         x = self.pool2(self.relu2(self.bn2(x)))
#         x = x.reshape(-1, 32 * 44)
#         feature = self.fc1_relu(self.fc1(x))
#         return [feature]
#
#     def get_sel_fea(self, x, plan=0):
#         if plan == 0:
#             x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
#             x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
#             x = x.reshape(-1, 32 * 44)
#             fealist = x
#         elif plan == 1:
#             x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
#             x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
#             x = x.reshape(-1, 32 * 44)
#             feature = self.fc1_relu(self.fc1(x))
#             fealist = feature
#         else:
#             fealist = []
#             x = self.conv1(x)
#             x = self.pool1(self.relu1(self.bn1(x)))
#             fealist.append(x.view(x.shape[0], -1))
#             x = self.conv2(x)
#             x = self.pool2(self.relu2(self.bn2(x)))
#             fealist.append(x.view(x.shape[0], -1))
#             x = x.reshape(-1, 32 * 44)
#             feature = self.fc1_relu(self.fc1(x))
#             fealist.append(feature)
#             fealist = torch.cat(fealist, dim=1)
#         return fealist
#

# class HARModel(nn.Module):
#     def __init__(self, n_feature=1024, out_dim=8):
#         super(HARModel, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1)
#         self.bn1 = nn.BatchNorm2d(1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(1)
#         self.conv2 = nn.Conv2d(
#             in_channels=1, out_channels=2, kernel_size=(3, 3), padding=1)
#         self.bn2 = nn.BatchNorm2d(2)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
#         self.fc1 = nn.Linear(in_features=1 * 1000, out_features=n_feature)
#         self.fc1_relu = nn.ReLU()
#         self.fc2 = nn.Linear(in_features=n_feature, out_features=out_dim)
#
#     def forward(self, x):
#         # print(x.shape)
#         x = self.conv1(x)
#         # print(x.shape)
#         x = self.pool1(self.relu1(self.bn1(x)))
#         # print(x.shape)
#         x = self.conv2(x)
#         x = self.pool2(self.relu2(self.bn2(x)))
#         # print(x.shape)
#         x = x.reshape(-1, 1 * 1000)
#         feature = self.fc1_relu(self.fc1(x))
#         out = self.fc2(feature)
#         print(out.shape)
#         return out
#
#     def getallfea(self, x):
#         fealist = []
#         x = self.conv1(x)
#         fealist.append(x.clone().detach())
#         x = self.pool1(self.relu1(self.bn1(x)))
#         x = self.conv2(x)
#         fealist.append(x.clone().detach())
#         return fealist
#
#     def getfinalfea(self, x):
#         x = self.conv1(x)
#         x = self.pool1(self.relu1(self.bn1(x)))
#         x = self.conv2(x)
#         x = self.pool2(self.relu2(self.bn2(x)))
#         x = x.reshape(-1, 32 * 44)
#         feature = self.fc1_relu(self.fc1(x))
#         return [feature]
#
#     def get_sel_fea(self, x, plan=0):
#         if plan == 0:
#             x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
#             x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
#             x = x.reshape(-1, 32 * 44)
#             fealist = x
#         elif plan == 1:
#             x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
#             x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
#             x = x.reshape(-1, 32 * 44)
#             feature = self.fc1_relu(self.fc1(x))
#             fealist = feature
#         else:
#             fealist = []
#             x = self.conv1(x)
#             x = self.pool1(self.relu1(self.bn1(x)))
#             fealist.append(x.view(x.shape[0], -1))
#             x = self.conv2(x)
#             x = self.pool2(self.relu2(self.bn2(x)))
#             fealist.append(x.view(x.shape[0], -1))
#             x = x.reshape(-1, 32 * 44)
#             feature = self.fc1_relu(self.fc1(x))
#             fealist.append(feature)
#             fealist = torch.cat(fealist, dim=1)
#         return fealist


class Baseline(nn.Module):
    def __init__(self, n_feature=1024, out_dim=8):
        super(Baseline, self).__init__()
        self.name = "baseline"
        self.c1, self.c2, self.c3 = 16, 32, 64
        self.mid_shape = 896
        # self.c1, self.c2, self.c3 = 64, 128, 256
        # self.mid_shape = 768

        self.conv1 = nn.Conv1d(
            in_channels=3, out_channels=self.c1, kernel_size=24)
        self.bn1 = nn.BatchNorm1d(self.c1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(
            in_channels=self.c1, out_channels=self.c2, kernel_size=16)
        self.bn2 = nn.BatchNorm1d(self.c2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.1)
        self.conv3 = nn.Conv1d(
            in_channels=self.c2, out_channels=self.c3, kernel_size=8)
        self.bn3 = nn.BatchNorm1d(self.c3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.1)

        self.pool = nn.MaxPool1d(self.c3)
        self.fc1 = nn.Linear(in_features=self.mid_shape, out_features=n_feature)
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=n_feature, out_features=out_dim)

        self.bns = [self.bn1, self.bn2, self.bn3]

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout1(self.relu1(self.bn1(x)))
        x = self.conv2(x)
        x = self.dropout2(self.relu2(self.bn2(x)))
        x = self.conv3(x)
        x = self.dropout3(self.relu3(self.bn3(x)))
        x = self.pool(x)
        # print(x.shape)
        x = x.reshape(-1, self.mid_shape)
        feature = self.fc1_relu(self.fc1(x))
        out = self.fc2(feature)
        # print(out.shape)
        return out

    def getallfea(self, x):
        fealist = []
        x = self.conv1(x)
        fealist.append(x.clone().detach())
        x = self.dropout1(self.relu1(self.bn1(x)))
        x = self.conv2(x)
        fealist.append(x.clone().detach())
        x = self.dropout2(self.relu2(self.bn2(x)))
        x = self.conv3(x)
        fealist.append(x.clone().detach())
        return fealist

    def getfinalfea(self, x):
        x = self.conv1(x)
        x = self.dropout1(self.relu1(self.bn1(x)))
        x = self.conv2(x)
        x = self.dropout2(self.relu2(self.bn2(x)))
        x = self.conv3(x)
        x = self.dropout3(self.relu3(self.bn3(x)))
        x = self.pool(x)
        x = x.reshape(-1, self.mid_shape)
        feature = self.fc1_relu(self.fc1(x))
        return [feature]


class BaselineEarlyExit(nn.Module):
    def __init__(self, n_feature=1024, out_dim=8):
        super(BaselineEarlyExit, self).__init__()
        self.name = "baseline"
        self.c1, self.c2, self.c3 = 16, 32, 64
        self.c = [16, 32, 64]
        self.mid_shape = 896
        num_hidden = 3
        
        # self.c1, self.c2, self.c3 = 64, 128, 256
        # self.mid_shape = 768
        self.complexity_factor = 1.2

        exit_hidden_sizes = [int(((self.complexity_factor ** 0.5) ** (num_hidden - idx)) * self.c[-1]) for idx in range(num_hidden)]

        self.conv1 = nn.Conv1d(
            in_channels=3, out_channels=self.c1, kernel_size=24)
        self.bn1 = nn.BatchNorm1d(self.c1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(
            in_channels=self.c1, out_channels=self.c2, kernel_size=16)
        self.bn2 = nn.BatchNorm1d(self.c2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.1)
        self.conv3 = nn.Conv1d(
            in_channels=self.c2, out_channels=self.c3, kernel_size=8)
        self.bn3 = nn.BatchNorm1d(self.c3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.1)

        self.pool = nn.MaxPool1d(self.c3)
        self.fc1 = nn.Linear(in_features=self.mid_shape, out_features=n_feature)
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=n_feature, out_features=out_dim)

        exit_blocks = []
        for block_idx in range(num_hidden):
            in_channels = self.c[block_idx]
            hidden = exit_hidden_sizes[block_idx]
            # exit_blocks += [ExitBlock(in_channels, in_channels*2, out_dim)]
            exit_blocks += [ExitBlock(in_channels, hidden, out_dim)]
        self.exit_blocks = nn.ModuleList(exit_blocks)

    def forward(self, x):
        out_blocks = []

        x = self.conv1(x)
        x = self.dropout1(self.relu1(self.bn1(x)))
        # print(x.shape)
        out_blocks += [x]

        x = self.conv2(x)
        x = self.dropout2(self.relu2(self.bn2(x)))
        # print(x.shape)
        out_blocks += [x]

        x = self.conv3(x)
        x = self.dropout3(self.relu3(self.bn3(x)))
        # print(x.shape)
        out_blocks += [x]

        out_exits = []
        for out_block, exit_block in zip(out_blocks, self.exit_blocks):
            out = exit_block(out_block)
            out_exits += [out]

        x = self.pool(x)
        # print(x.shape)
        x = x.reshape(-1, self.mid_shape)
        # print(x.shape)
        feature = self.fc1_relu(self.fc1(x))
        out = self.fc2(feature)

        out = torch.stack(out_exits + [out], dim=1)
        # print(out.shape)
        return out

    def getallfea(self, x):
        fealist = []
        x = self.conv1(x)
        fealist.append(x.clone().detach())
        x = self.dropout1(self.relu1(self.bn1(x)))
        x = self.conv2(x)
        fealist.append(x.clone().detach())
        x = self.dropout2(self.relu2(self.bn2(x)))
        x = self.conv3(x)
        fealist.append(x.clone().detach())
        return fealist


class BaselineEarlyExitMetaModel(nn.Module):
    def __init__(self, base_model, n_feature=1024, out_dim=8):
        super(BaselineEarlyExitMetaModel, self).__init__()
        self.c = [16, 32, 64]

        self.base_model = base_model
        
        exit_blocks = []
        for block_idx in range(3):
            in_channels = self.c[block_idx]
            exit_blocks += [ExitBlock(in_channels, in_channels*2, out_dim)]
        self.exit_blocks = nn.ModuleList(exit_blocks)
        # self.fc_ee = nn.Linear(n_feature * 3, out_dim)

    def forward(self, x):
        out_blocks = []
        b = self.base_model
        x = b.conv1(x)
        x = b.dropout1(b.relu1(b.bn1(x)))
        # print(x.shape)
        out_blocks += [x]

        x = b.conv2(x)
        x = b.dropout2(b.relu2(b.bn2(x)))
        # print(x.shape)
        out_blocks += [x]

        x = b.conv3(x)
        x = b.dropout3(b.relu3(b.bn3(x)))
        # print(x.shape)
        out_blocks += [x]

        out_exits = []
        for out_block, exit_block in zip(out_blocks, self.exit_blocks):
            out = exit_block(out_block)
            out_exits += [out]

        # x = b.pool(x)
        # # print(x.shape)
        # x = x.reshape(-1, b.mid_shape)
        # feature = b.fc1_relu(b.fc1(x))
        # out = b.fc2(feature)

        out = torch.stack(out_exits + [out], dim=1)
        # out = torch.stack(out_exits, dim=1)
        # print(out.shape)
        return out




class ExitWeightedLoss:

    def __init__(self, alpha = [1, 1, 1, 1]):
        self.alpha=torch.tensor(alpha)

    def __call__(self, logits, labels, gamma = [1, 1, 1, 1]):

        batch_size, num_exits, _ = logits.shape
        labels = labels.long()
        loss = 0.0
        for ex in range(num_exits):
            exit_logits = logits[:, ex, :]
            # print(exit_logits, labels)
            loss += self.alpha[ex] * gamma[ex] * F.cross_entropy(exit_logits, labels)

        return loss


def metric_earlyexit(logits, labels, ensemble_weights=torch.tensor([1, 1, 1, 1]), average="weighted"):
    _, num_exits, num_classes = logits.shape
    scale = ensemble_weights.sum()

    pred_labels = logits.softmax(dim=-1).mul(ensemble_weights).sum(dim=-2).div(scale).argmax(-1)

    f1 = tm.f1(pred_labels, labels, num_classes=num_classes, average=average)

    return f1




# def try_train():
#     from util.traineval import train
#     from uncertainty import evaluate_model_on_tests
#     device= "cuda" if torch.cuda.is_available() else "cpu"
#     model = Baseline()
#     # lossfunc = nn.CrossEntropyLoss()
#     lossfunc = BaselineLoss()
#     for i in range(7,8):
#         print("pid", i)
#         train_dataloader = get_dataloader(center=0, train=True)
#         test_dataloader = get_dataloader(center=i, train=False)

#         optimizer = Optimizer(model.parameters(), lr=LR)
#         for epoch in range(NUM_EPOCHS_POOLED):
#             train(model, train_dataloader, optimizer, lossfunc, device)
#             res = evaluate_model_on_tests(model, [train_dataloader], metric)
#             print("train acc", res)
#             if epoch % 5 == 4:
#                 res = evaluate_model_on_tests(model, [test_dataloader], metric)
#                 print("test acc", res)


def try_train(earlyexit=False):
    print("har_centralized_earlyexit_" + str(earlyexit))
    from util.traineval import train
    from uncertainty import evaluate_model_on_tests, evaluate
    device= "cuda" if torch.cuda.is_available() else "cpu"
    
    if earlyexit:
        print("early exit")
        model = BaselineEarlyExit()
        # model = BaselineEarlyExitP()
        lossfunc = ExitWeightedLoss()
    else:
        print("single exit")
        model = Baseline()
        lossfunc = BaselineLoss()
    for i in range(1):
        print("pid", i)
        train_dataloader = get_dataloader(center=0, train=True, pooled=True)
        test_dataloader = get_dataloader(center=i, train=False, pooled=True)
        # print(train_dataloader.dataset.__len__())
        # print(test_dataloader.dataset.__len__())

        optimizer = Optimizer(model.parameters(), lr=LR)
        for epoch in range(60):
            print("epoch", epoch, end=", ")
            train(model, train_dataloader, optimizer, lossfunc, device)
            res = evaluate_model_on_tests(model, [train_dataloader], metric, earlyexit=earlyexit)
            print("train acc", res['client_test_0'], end=", ")
            res = evaluate_model_on_tests(model, [test_dataloader], metric, earlyexit=earlyexit)
            print("test acc", res['client_test_0'])

        dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(model,  [test_dataloader], metric, MCDO=False, return_pred=True, earlyexit=earlyexit)
        evaluate(dict_cindex, y_true, y_pred, variance, entropy, uncertainty=False, id="har_centralized_earlyexit" + str(earlyexit), task="multiclass")


if __name__ == '__main__':
    # tmp_har()
    # try_train(True)
    # try_train(False)
    # get_dataloader()
    get_dataloader_oppo()