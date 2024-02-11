import torch
import torch.nn as nn
import torch.nn.functional as func
from collections import OrderedDict, Counter
import numpy as np
from sklearn import metrics
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score

from torch.utils import data
import numpy as np
import glob
import gc
import librosa
import matplotlib.pyplot as plt
import pandas as pd

import torchvision
from torchvision.io import read_image
import torchvision.transforms as T

# from scipy.io import wavfile
# from scipy.io import loadmat
from scipy import signal

# from earlyexit import ExitBlock, ExitBlock2D, ExitBlockAdaptive


NUM_CLIENTS = 6
BATCH_SIZE = 64
NUM_EPOCHS_POOLED = 20
NUM_EPOCHS_CENTRALIZED = 40
LR = 1e-4 #1e-4, 5e-5, 1e-5
Optimizer = torch.optim.Adam

test_split_proportion = 0.4 #0.3333


# class BaselineLoss(_Loss):
#     def __init__(self):
#         super(BaselineLoss, self).__init__()
#         self.bce = torch.nn.BCELoss()

#     def forward(self, input: torch.Tensor, target: torch.Tensor):
#         return self.bce(input, target)

# def metric(y_true, y_pred):
#     y_true = y_true.astype("uint8")
#     # The try except is needed because when the metric is batched some batches
#     # have one class only
#     try:
#         # return roc_auc_score(y_true, y_pred)
#         # proposed modification in order to get a metric that calcs on center 2
#         # (y=1 only on that center)
#         return ((y_pred > 0.5) == y_true).mean()
#     except ValueError:
#         return np.nan

def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    # return (8717 // NUM_CLIENTS // batch_size) * NUM_EPOCHS_CENTRALIZED // num_updates
    
    return (6505 // NUM_CLIENTS // batch_size) * NUM_EPOCHS_CENTRALIZED // num_updates


"""====================== resnet =========================="""

def init_weights(model):

    for module in model.modules():

        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):

    name = "res_net_18"

    def __init__(self, out_channels=1, seed=None):
        super().__init__()
        self.name = "baseline"
        self.out_channels = out_channels
        self.seed = seed

        self.hidden_sizes = [64, 128, 256, 512]
        self.layers = [2, 2, 2, 2]
        self.strides = [1, 2, 2, 2]
        self.inplanes = self.hidden_sizes[0]
        self.drop_after = [1, 3, 5, 7]

        in_block = [nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)]
        in_block += [nn.BatchNorm2d(self.inplanes)]
        in_block += [nn.ReLU(inplace=True)]
        in_block += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        self.in_block = nn.Sequential(*in_block)

        blocks = []
        for h, l, s in zip(self.hidden_sizes, self.layers, self.strides):
            blocks += [self._make_layer(h, l, s)]
        self.blocks = nn.Sequential(*blocks)

        # # adding dropouts
        # for block_idx in range(len(self.blocks)):            
        #     self.blocks[block_idx].add_module("dropout", MCDropout(0.2))

        out_block = [nn.AdaptiveAvgPool2d(1)]
        out_block += [nn.Flatten(1)]
        out_block += [nn.Linear(self.hidden_sizes[-1], self.out_channels)]
        self.out_block = nn.Sequential(*out_block)

        if self.seed is not None:
            torch.manual_seed(seed)

        self.apply(init_weights)

    def _make_layer(self, planes, blocks, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(_conv1x1(self.inplanes, planes, stride), nn.BatchNorm2d(planes))

        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes

        for _ in range(1, blocks):
            layers += [BasicBlock(self.inplanes, planes)]
        
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.in_block(x)
        x = self.blocks(x)
        x = self.out_block(x)
        x = torch.sigmoid(x)

        return x
    
    def getallfea(self, x):
        fealist = []
        fea1 = self.in_block[0](x).clone().detach()
        fealist.append(fea1)

        fea = self.in_block(x)
        for basic_block in self.blocks:
            # for layer in block:
            #     name = str(layer)
            #     print(name)
            #     if name.startswith("BatchNorm"):
            #         fealist.append(fea.clone().detach())
            #     fea = l(fea)
            identity = fea
            fea = basic_block.conv1(fea)

            fealist.append(fea.clone().detach())
            fea = basic_block.bn1(fea)
            fea = basic_block.relu(fea)
            fea = basic_block.conv2(fea)

            fealist.append(fea.clone().detach())
            fea = basic_block.bn2(fea)

            if basic_block.downsample is not None:
                identity = basic_block.downsample[0](identity)
                fealist.append(identity.clone().detach())
                identity = basic_block.downsample[1](identity)

            fea += identity
            fea = basic_block.relu(fea)

        return fealist



class ExitBlock(nn.Module):

    def __init__(self, in_channels, hidden_sizes, out_channels):
        super().__init__()

        layers = [nn.AdaptiveAvgPool2d(1)]
        layers += [nn.Flatten(1)]
        layers += [nn.Linear(in_channels, hidden_sizes)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(hidden_sizes, out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        return self.layers(x)


class BaselineEarlyExit(ResNet18):
    
    name = "res_net_18_early_exit"

    def __init__(self, *args, exit_after=[1,3,5,7], complexity_factor=1.2, **kwargs):
        self.exit_after = exit_after
        self.complexity_factor = complexity_factor

        super().__init__(*args, **kwargs)

        to_exit = [2, 8, 15, 24, 31, 40, 47, 56]
        hidden_sizes = len(self.hidden_sizes)

        num_hidden = len(self.hidden_sizes)
        exit_hidden_sizes = [int(((self.complexity_factor ** 0.5) ** (num_hidden - idx)) * self.hidden_sizes[-1]) for idx in range(num_hidden)]
        exit_hidden_sizes = [h for pair in zip(exit_hidden_sizes, exit_hidden_sizes) for h in pair]

        if self.exit_after == -1:
            self.exit_after = range(len(to_exit))

        num_exits = len(to_exit)

        if (len(self.exit_after) > num_exits) or not set(self.exit_after).issubset(list(range(num_exits))):
            raise ValueError("valid exit points: {}".format(", ".join(str(n) for n in range(num_exits))))

        self.exit_hidden_sizes = np.array(exit_hidden_sizes)[self.exit_after]

        blocks = []
        for idx, module in enumerate(self.blocks.modules()):
            if idx in to_exit:
                blocks += [module]
        self.blocks = nn.ModuleList(blocks)
        # print(self.blocks)
        
        # adding dropout
        for block_idx in self.exit_after:            
            self.blocks[block_idx].add_module("dropout", MCDropout(0.2))
        # print(self.blocks)
        
        idx = 0
        exit_blocks = []
        for block_idx, block in enumerate(self.blocks):
            if block_idx in self.exit_after:
                in_channels = block.conv1.out_channels
                exit_blocks += [ExitBlock(in_channels, self.exit_hidden_sizes[idx], self.out_channels)]
                print(block_idx, in_channels, self.exit_hidden_sizes[idx], self.out_channels)
                idx += 1
        self.exit_blocks = nn.ModuleList(exit_blocks)

        self.apply(init_weights)

    def forward(self, x):

        out = self.in_block(x)

        out_blocks = []
        for block in self.blocks:
            out = block(out)
            out_blocks += [out]

        out_exits = []
        for exit_after, exit_block in zip(self.exit_after, self.exit_blocks):
            out = exit_block(out_blocks[exit_after])
            out = torch.sigmoid(out)
            out_exits += [out]

        out = self.out_block(out_blocks[-1])
        out = torch.sigmoid(out)
        out = torch.stack(out_exits + [out], dim=1)

        return out


class MCDropout(nn.Dropout):
    # def __init__(self, drop_prob=0.2):
        # self.dropout = nn.Dropout(p=drop_prob)
        # self.p = drop_prob

    def forward(self, x):
        print("self.p = ", self.p)
        out = F.dropout(x, self.p, True, self.inplace)
        # out = self.dropout(x)
        print(x - out)
        return out


class Baseline(BaselineEarlyExit):
    
    name = "res_net_18_mc_drop"

    def __init__(self, *args, drop_after=[1,3,5,7], drop_prob=0.2, **kwargs):
        self.drop_after = drop_after
        self.drop_prob = drop_prob

        super().__init__(*args, exit_after=drop_after, **kwargs)

        self.drop_after = self.exit_after

        self.__delattr__("exit_after")
        self.__delattr__("exit_blocks")

        # # adding dropout
        # for block_idx in self.drop_after:            
        #     self.blocks[block_idx].add_module("dropout", MCDropout(self.drop_prob))
        
        self.dropout = nn.Dropout(p=drop_prob)


    def forward(self, x):

        x = self.in_block(x)
        # x = self.blocks(x)
        for idx, block in enumerate(self.blocks):
            # print(idx)
            x = block(x)
            if idx in self.drop_after:
                x = self.dropout(x)
        x = self.out_block(x)
        x = torch.sigmoid(x)

        return x

"""====================== resnet end ======================"""


# class BaselineLoss(_Loss):
#     """Weighted focal loss
#     See this [link](https://amaarora.github.io/2020/06/29/FocalLoss.html) for
#     a good explanation
#     Attributes
#     ----------
#     alpha: torch.tensor of size 8, class weights
#     gamma: torch.tensor of size 1, positive float, for gamma = 0 focal loss is
#     the same as CE loss, increases gamma reduces the loss for the "hard to classify
#     examples"
#     """

#     def __init__(
#             self,
#             alpha=torch.tensor(
#                 [1,1]
#             ),
#             gamma=2.0,
#     ):
#         super(BaselineLoss, self).__init__()
#         self.alpha = alpha.to(torch.float)
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         """Weighted focal loss function
#         Parameters
#         ----------
#         inputs : torch.tensor of size 8, logits output by the model (pre-softmax)
#         targets : torch.tensor of size 1, int between 0 and 7, groundtruth class
#         """
#         targets = targets.view(-1, 1).type_as(inputs)
#         logpt = F.log_softmax(inputs, dim=1)
#         logpt = logpt.gather(1, targets.long())
#         logpt = logpt.view(-1)
#         pt = logpt.exp()
#         self.alpha = self.alpha.to(targets.device)
#         at = self.alpha.gather(0, targets.data.view(-1).long())
#         logpt = logpt * at
#         loss = -1 * (1 - pt) ** self.gamma * logpt

#         return loss.mean()


# def metric(y_true, logits):
#     y_true = y_true.reshape(-1)
#     preds = np.argmax(logits, axis=1)
#     # return metrics.balanced_accuracy_score(y_true, preds)
#     return metrics.accuracy_score(y_true, preds)
#     # return metrics.f1_score(y_true, preds, average='macro')


class BaselineLoss(_Loss):
    def __init__(self):
        super(BaselineLoss, self).__init__()
        self.bce = torch.nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # print(pred.shape, target.shape)
        # print(pred[:5], target[:5])
        # print("loss", self.bce(pred, target))
        return self.bce(pred, target)


def metric(y_true, y_pred):
    # print(y_true.shape, y_pred.shape)
    y_true = y_true.astype("uint8")
    # The try except is needed because when the metric is batched some batches
    # have one class only
    try:
        # return roc_auc_score(y_true, y_pred)
        # proposed modification in order to get a metric that calcs on center 2
        # (y=1 only on that center)
        return ((y_pred > 0.5) == y_true).mean()
    except ValueError:
        # print(y_true, y_pred)
        # print("Error: all", y_true[0])
        # return ((y_pred > 0.5) == y_true).mean()
        return np.nan


# def metric(y_true, y_pred):
#     # print(y_true.tolist(), y_pred.tolist())
#     # return roc_auc_score(y_true, y_pred)
#     # y_true = y_true.reshape(-1)
#     # preds = np.argmax(logits)
#     y_pred = np.round(y_pred)
#     return recall_score(y_true, y_pred, average='macro')
#     # # return metrics.balanced_accuracy_score(y_true, y_pred)
#     # return metrics.accuracy_score(y_true, y_pred)
#     # return metrics.f1_score(y_true, y_pred)


class DataReader:
    label_map = {"Normal": 0, "Abnormal": 1}
    @staticmethod
    def read_label(file_name, from_file=True):
        """Function saves information about the patient from header file in this order:
        sampling frequency, length of the signal, voltage resolution, age, sex, list of diagnostic labels both in
        SNOMED and abbreviations (sepate lists)"""
        if from_file:
            lines=[]
            with open(file_name, "r") as file:
                for line_idx, line in enumerate(file):
                    lines.append(line)
        else:
            lines=file_name
        label = lines[-1][2:-1]
        return DataReader.label_map[label]
    
    def read_content(content):
        return np.array(content["image_name"]), np.array(content["file_label"])


def build_spectogram(file_audio_series, sr, sound_file_name, image_folder, audio=None):
    plt.interactive(False)
    spec_image = plt.figure(figsize=[0.72,0.72])
    ax = spec_image.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    
    # normalisation
    if audio is not None:
        file_audio_series=file_audio_series/ (np.sqrt(np.sum(audio*audio)/len(audio)))

    #####################################################################
    # Edit here to create the log melspectrogram and save it as a jpg file in google drive
    spectogram = librosa.feature.melspectrogram(y=file_audio_series, sr=sr) # win_length=, 
    fig = librosa.display.specshow(librosa.power_to_db(spectogram, ref=np.max), y_axis='mel', x_axis='time', sr=sr, ax=ax) # window_length=2048, hop_length=2048
    image_filepath = image_folder + sound_file_name +'.jpg'
    #####################################################################
    
    plt.savefig(image_filepath, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    spec_image.clf()
    plt.close(spec_image)
    plt.close('all')
    del sound_file_name, file_audio_series, sr, spec_image, ax, spectogram


def process_spectogram():
    segment_dur_secs = 5
    sr = 2000
    segment_length = sr * segment_dur_secs
    
    # data_dir = "datautil/PCG/"
    # data_dir = "~/rds/hpc-work/data_util/PCG/" // don't work when running
    data_dir = "/home/yz798/rds/hpc-work/data_util/PCG/"
    datasets = ["training-" + idx + "/" for idx in "abcdef"]
    for dataset in datasets[1:]:
        contents = []
        input_directory = data_dir + dataset
        file_list = glob.glob(input_directory + "*.wav", recursive=True)
        num_files = len(file_list)
        print(num_files)
        
        for filename in file_list:
            sfile_name = filename.split('/')[-1].split('.')[0]
            header_file_name = filename[:-3] + "hea"
            file_label = DataReader.read_label(header_file_name)
            y, _ = librosa.load(filename, sr=sr)
            # print(librosa.get_duration(y=y, sr=sr))
            num_sections = int(np.floor(len(y) / segment_length))
            # print(num_sections)
            for i in range(num_sections):
                t = y[i * segment_length: (i + 1) * segment_length]
                image_file_name = sfile_name + "_" + str(i) + "_normed"
                build_spectogram(t, sr, image_file_name, input_directory, audio=y)
                contents.append(np.array([sfile_name, input_directory + image_file_name, file_label], dtype=str))
        
        
        np.savetxt(input_directory + "contents_normed.csv", contents, delimiter=",", fmt='%s')
        gc.collect()



def get_dataloader(center: int = 0,
                   train: bool = True,
                   pooled: bool = False):
    sr = 2000
    
    # data_dir = "datautil/PCG/"
    # data_dir = "~/rds/hpc-work/data_util/PCG/" // don't work when running
    data_dir = "/home/yz798/rds/hpc-work/data_util/PCG/"
    datasets = ["training-" + idx + "/" for idx in "abcdef"]
    if pooled:
        X, Y = [], []
        for dataset in datasets:
            input_directory = data_dir + dataset
            
            # # rgb images
            # contents = pd.read_csv(input_directory + "contents_normed.csv", delimiter=",", names=["fileID", "image_name", "file_label"], header=None)
            # X1, Y1 = DataReader.read_content(contents)
            
            # matrices
            X1 = np.load(input_directory + "X.npy")
            Y1 = np.load(input_directory + "Y.npy")
            
            train_x1, test_x1, train_y1, test_y1 = train_test_split(X1, Y1, test_size=test_split_proportion, random_state=42)
            if train:
                X1, Y1 = train_x1, train_y1
            else:
                X1, Y1 = test_x1, test_y1
            
            X.append(X1); Y.append(Y1)
        X = np.concatenate(X); Y = np.concatenate(Y)
    else:
        input_directory = data_dir + datasets[center]
        
        # contents = pd.read_csv(input_directory + "contents_normed.csv", delimiter=",", names=["fileID", "image_name", "file_label"], header=None)
        # X, Y = DataReader.read_content(contents)
        
        X = np.load(input_directory + "X.npy")
        Y = np.load(input_directory + "Y.npy")

        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=test_split_proportion, random_state=42)

        if train:
            X, Y = train_x, train_y
        else:
            X, Y = test_x, test_y
        
    
    
    # classes = Counter(list(train_y))
    # print("train dist", dict(classes))
    # classes = Counter(list(test_y))
    # print("test dist", dict(classes))
    
    
        
    # transform = T.Resize((64,64))
    # def get_image_input(image_name):
    #     sound_file_name = image_name +'.jpg'
    #     img = read_image(sound_file_name)
    #     img = transform(img)
    #     # img = img.resize((64, 64))
    #     # img_array = img_to_array(img)
    #     # print(img.size())
    #     return img.numpy()
    # imgs = []
    # for x in X:
    #     imgs.append(get_image_input(x))
    # X = np.array(imgs)
    # print(X.shape, Y.shape)
    
    Y = np.expand_dims(Y, axis=1)
    
    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(Y)
    
    tensor_y = tensor_y.view(-1, 1)
    
    # print(center, tensor_x.shape, tensor_y.shape)

    data = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)

    return data_loader


def tmp_pcg():

    get_dataloader(center=0, train=True)
    get_dataloader(center=0, train=False)


# 4-layer

# class Baseline(nn.Module):
#     def __init__(self, n_feature=512, out_dim=1):
#         super(Baseline, self).__init__()
#         self.c1, self.c2, self.c3, self.c4 = 32, 64, 64, 32
#         self.mid_shape = 128
        
#         self.conv1 = nn.Conv2d(
#             in_channels=3, out_channels=self.c1, kernel_size=(3, 3))
#         self.bn1 = nn.BatchNorm2d(self.c1)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(p=0.1)
#         self.pool1 = nn.MaxPool2d((2, 2))
        
#         self.conv2 = nn.Conv2d(
#             in_channels=self.c1, out_channels=self.c2, kernel_size=(3, 3))
#         self.bn2 = nn.BatchNorm2d(self.c2)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(p=0.1)
#         self.pool2 = nn.MaxPool2d((2, 2))
        
#         self.conv3 = nn.Conv2d(
#             in_channels=self.c2, out_channels=self.c3, kernel_size=(3, 3))
#         self.bn3 = nn.BatchNorm2d(self.c3)
#         self.relu3 = nn.ReLU()
#         self.dropout3 = nn.Dropout(p=0.1)
#         self.pool3 = nn.MaxPool2d((2, 2))
        
#         self.conv4 = nn.Conv2d(
#             in_channels=self.c2, out_channels=self.c4, kernel_size=(3, 3))
#         self.bn4 = nn.BatchNorm2d(self.c4)
#         self.relu4 = nn.ReLU()
#         self.dropout4 = nn.Dropout(p=0.1)
#         self.pool4 = nn.MaxPool2d((2, 2))

#         self.fc1 = nn.Linear(in_features=self.mid_shape, out_features=n_feature)
#         self.fc1_relu = nn.ReLU()
#         self.fc2 = nn.Linear(in_features=n_feature, out_features=out_dim)
        

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool1(self.dropout1(self.relu1(self.bn1(x))))
#         x = self.conv2(x)
#         x = self.pool2(self.dropout2(self.relu2(self.bn2(x))))
#         x = self.conv3(x)
#         x = self.pool3(self.dropout3(self.relu3(self.bn3(x))))
#         x = self.conv4(x)
#         x = self.pool4(self.dropout4(self.relu4(self.bn4(x))))
#         # print(x.shape)
#         x = x.reshape(-1, self.mid_shape)
#         feature = self.fc1_relu(self.fc1(x))
#         out = self.fc2(feature)
#         # print(out.shape)
#         out = torch.sigmoid(out)
#         return out

#     def getallfea(self, x):
#         fealist = []
#         x = self.conv1(x)
#         fealist.append(x.clone().detach())
#         x = self.pool1(self.dropout1(self.relu1(self.bn1(x))))
#         x = self.conv2(x)
#         fealist.append(x.clone().detach())
#         x = self.pool2(self.dropout2(self.relu2(self.bn2(x))))
#         x = self.conv3(x)
#         fealist.append(x.clone().detach())
#         x = self.pool3(self.dropout3(self.relu3(self.bn3(x))))
#         x = self.conv4(x)
#         fealist.append(x.clone().detach())
#         return fealist

# # 3-layer
# class Baseline(nn.Module):
#     def __init__(self, n_feature=64, out_dim=1):
#         super(Baseline, self).__init__()
#         self.c1, self.c2, self.c3 = 64, 64, 32
#         self.mid_shape = 512
#         # self.mid_shape = 3584
#         # self.conv1 = nn.Conv2d(
#         #     in_channels=1, out_channels=self.c1, kernel_size=(11, 11), stride=(6,6))
#         self.conv1 = nn.Conv2d(
#             in_channels=1, out_channels=self.c1, kernel_size=(6, 6), stride=(2,2))
#         self.bn1 = nn.BatchNorm2d(self.c1)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(p=0.2)
        
#         self.pool1 = nn.MaxPool2d((2, 2))
        
#         self.conv2 = nn.Conv2d(
#             in_channels=self.c1, out_channels=self.c2, kernel_size=(3, 3))
#         self.bn2 = nn.BatchNorm2d(self.c2)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(p=0.2)
        
#         self.pool2 = nn.MaxPool2d((2, 2))
        
        
#         self.conv3 = nn.Conv2d(
#             in_channels=self.c2, out_channels=self.c3, kernel_size=(3, 3))
#         self.bn3 = nn.BatchNorm2d(self.c3)
#         self.relu3 = nn.ReLU()
#         self.dropout3 = nn.Dropout(p=0.2)
        
#         self.bns = [self.bn1, self.bn2, self.bn3]
        
#         self.pool3 = nn.MaxPool2d((2, 2))

#         self.fc1 = nn.Linear(in_features=self.mid_shape, out_features=n_feature)
#         self.fc1_relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(in_features=n_feature, out_features=out_dim)
        

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool1(self.dropout1(self.relu1(self.bn1(x))))
#         x = self.conv2(x)
#         x = self.pool2(self.dropout2(self.relu2(self.bn2(x))))
#         x = self.conv3(x)
#         x = self.pool3(self.dropout3(self.relu3(self.bn3(x))))
#         # print(x.shape)
        
#         x = x.reshape(-1, self.mid_shape)
        
        
#         feature = self.fc1_relu(self.fc1(x))
#         feature = self.dropout(feature)
#         out = self.fc2(feature)
#         # print(out.shape)
#         out = torch.sigmoid(out)
#         # out = self.softmax(out)
#         return out

#     def getallfea(self, x):
#         fealist = []
#         x = self.conv1(x)
#         fealist.append(x.clone().detach())
#         return fealist



# class Baseline(nn.Module):
#     """
#     nilanon et al. model
#     """
#     def __init__(self, n_feature=64, out_dim=1):
#         super(Baseline, self).__init__()
#         self.c1, self.c2, self.c3 = 64, 64, 32
#         self.mid_shape = 3584
#         self.conv1 = nn.Conv2d(
#             in_channels=1, out_channels=self.c1, kernel_size=(11, 11), stride=(6,6))

#         self.bn1 = nn.BatchNorm2d(self.c1)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(p=0.2)
        
#         self.pool1 = nn.MaxPool2d((2, 2))
        
#         self.fc1 = nn.Linear(in_features=self.mid_shape, out_features=n_feature)
#         self.fc1_relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(in_features=n_feature, out_features=out_dim)
        
#         self.bns = [self.bn1]
        

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool1(self.dropout1(self.relu1(self.bn1(x))))
#         # print(x.shape)
        
#         x = x.reshape(-1, self.mid_shape)
        
        
#         feature = self.fc1_relu(self.fc1(x))
#         feature = self.dropout(feature)
#         out = self.fc2(feature)
#         # print(out.shape)
#         out = torch.sigmoid(out)
#         # out = self.softmax(out)
#         return out

#     def getallfea(self, x):
#         fealist = []
#         x = self.conv1(x)
#         fealist.append(x.clone().detach())
#         return fealist


# class Baseline(nn.Module):
#     """
#     nilanon et al. input, three-layer model
#     """
#     def __init__(self, n_feature=512, out_dim=1):
#         super(Baseline, self).__init__()
#         # # this one works but a bit time-consuming to train
#         # self.c1, self.c2, self.c3 = 32, 64, 128
#         # self.mid_shape = 12800
        
#         self.c1, self.c2, self.c3 = 16, 32, 64
#         self.mid_shape = 6400
        
        
#         self.conv1 = nn.Conv2d(
#             in_channels=1, out_channels=self.c1, kernel_size=(3, 3))
#         self.bn1 = nn.BatchNorm2d(self.c1)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(p=0.1)
        
#         self.pool1 = nn.MaxPool2d((2, 2))
        
#         self.conv2 = nn.Conv2d(
#             in_channels=self.c1, out_channels=self.c2, kernel_size=(3, 3))
#         self.bn2 = nn.BatchNorm2d(self.c2)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(p=0.1)
        
#         self.pool2 = nn.MaxPool2d((2, 2))
        
#         self.conv3 = nn.Conv2d(
#             in_channels=self.c2, out_channels=self.c3, kernel_size=(3, 3))
#         self.bn3 = nn.BatchNorm2d(self.c3)
#         self.relu3 = nn.ReLU()
#         self.dropout3 = nn.Dropout(p=0.1)
        
#         self.bns = [self.bn1, self.bn2, self.bn3]
        
#         self.pool3 = nn.MaxPool2d((2, 2))

#         self.fc1 = nn.Linear(in_features=self.mid_shape, out_features=n_feature)
#         self.fc1_relu = nn.ReLU()
#         # self.dropout = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(in_features=n_feature, out_features=out_dim)
        

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool1(self.dropout1(self.relu1(self.bn1(x))))
#         x = self.conv2(x)
#         x = self.pool2(self.dropout2(self.relu2(self.bn2(x))))
#         x = self.conv3(x)
#         x = self.pool3(self.dropout3(self.relu3(self.bn3(x))))
#         # print(x.shape)
        
#         x = x.reshape(-1, self.mid_shape)
        
        
#         feature = self.fc1_relu(self.fc1(x))
#         # feature = self.dropout(feature)
#         out = self.fc2(feature)
#         # print(out.shape)
#         out = torch.sigmoid(out)
#         # out = self.softmax(out)
#         return out

#     def getallfea(self, x):
#         fealist = []
#         x = self.conv1(x)
#         fealist.append(x.clone().detach())
#         x = self.pool1(self.dropout1(self.relu1(self.bn1(x))))
#         x = self.conv2(x)
#         fealist.append(x.clone().detach())
#         x = self.pool2(self.dropout2(self.relu2(self.bn2(x))))
#         x = self.conv3(x)
#         fealist.append(x.clone().detach())

#         return fealist

# # from audio_res_net_18 import ResNet18EarlyExit as BaselineEarlyExit
# class BaselineEarlyExit(nn.Module):
#     """
#     nilanon et al. input, three-layer model
#     """
#     def __init__(self, n_feature=512, out_dim=1):
#         super(BaselineEarlyExit, self).__init__()
#         self.c1, self.c2, self.c3 = 16, 32, 64
#         self.c = [16, 32, 64]
#         self.pool_dim = [4, 2, 1]
#         self.mid_shape = 6400
        
        
#         self.conv1 = nn.Conv2d(
#             in_channels=1, out_channels=self.c1, kernel_size=(3, 3))
#         self.bn1 = nn.BatchNorm2d(self.c1)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(p=0.1)
        
#         self.pool1 = nn.MaxPool2d((2, 2))
        
#         self.conv2 = nn.Conv2d(
#             in_channels=self.c1, out_channels=self.c2, kernel_size=(3, 3))
#         self.bn2 = nn.BatchNorm2d(self.c2)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(p=0.1)
        
#         self.pool2 = nn.MaxPool2d((2, 2))
        
#         self.conv3 = nn.Conv2d(
#             in_channels=self.c2, out_channels=self.c3, kernel_size=(3, 3))
#         self.bn3 = nn.BatchNorm2d(self.c3)
#         self.relu3 = nn.ReLU()
#         self.dropout3 = nn.Dropout(p=0.1)
        
#         self.bns = [self.bn1, self.bn2, self.bn3]
        
#         self.pool3 = nn.MaxPool2d((2, 2))

#         self.fc1 = nn.Linear(in_features=self.mid_shape, out_features=n_feature)
#         self.fc1_relu = nn.ReLU()
#         # self.dropout = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(in_features=n_feature, out_features=out_dim)
        
#         exit_blocks = []
#         for block_idx in range(3):
#             in_channels = self.c[block_idx]
#             dim = self.pool_dim[block_idx]
#             # exit_blocks += [ExitBlock2D(in_channels, int(in_channels * 2 / 3.0 + 8), out_dim, binary=True)]
            
#             # learning_factor = 0.2
#             # hidden =  int((1+learning_factor) ** ((3 - block_idx) / 2.0 ) * 512)
            
#             # hidden = int(in_channels * 2 / 3.0 + 2)
#             hidden = int(in_channels * dim * 2 / 3.0 + 2)
#             print(hidden)
#             # exit_blocks += [ExitBlock2D(in_channels, hidden, out_dim)] #int(in_channels * dim * 2 / 3.0 + 2), 
#             exit_blocks += [ExitBlockAdaptive(in_channels, dim, hidden, out_dim)]
                                              
#         self.exit_blocks = nn.ModuleList(exit_blocks)
#         self.flatten = nn.Flatten(2)
        

#     def forward(self, x):
#         out_blocks = []
#         x = self.conv1(x)
#         x = self.dropout1(self.relu1(self.bn1(x)))
#         out_blocks += [x]
#         x = self.pool1(x)
        
#         x = self.conv2(x)
#         x = self.dropout2(self.relu2(self.bn2(x)))
#         out_blocks += [x]
#         x = self.pool2(x)
        
#         x = self.conv3(x)
#         x = self.dropout3(self.relu3(self.bn3(x)))
#         out_blocks += [x]
#         x = self.pool3(x)
#         # print(x.shape)
        
#         out_exits = []
#         for out_block, exit_block in zip(out_blocks, self.exit_blocks):
#             out = exit_block(out_block)
#             out = torch.sigmoid(out)
#             out_exits += [out]
        
#         x = x.reshape(-1, self.mid_shape)
        
        
#         feature = self.fc1_relu(self.fc1(x))
#         # feature = self.dropout(feature)
#         out = self.fc2(feature)
#         # print(out.shape)
#         out = torch.sigmoid(out)
#         # out = self.softmax(out)
        
#         out = torch.stack(out_exits + [out], dim=1)
#         # print(out)
#         return out

#     def getallfea(self, x):
#         fealist = []
#         x = self.conv1(x)
#         fealist.append(x.clone().detach())
#         x = self.pool1(self.dropout1(self.relu1(self.bn1(x))))
#         x = self.conv2(x)
#         fealist.append(x.clone().detach())
        
#         x = self.pool2(self.dropout2(self.relu2(self.bn2(x))))
#         x = self.conv3(x)
#         fealist.append(x.clone().detach())
        
#         return fealist


class ExitWeightedLoss:

    def __init__(self, alpha = [1, 1, 1, 1, 1]):
        self.alpha=torch.tensor(alpha)
        self.bce = BaselineLoss() #torch.nn.BCELoss()

    def __call__(self, logits, labels, gamma = [1, 1, 1, 1, 1]):

        batch_size, num_exits, _ = logits.shape
        # labels = labels.long()
        loss = 0.0
        for ex in range(num_exits):
            exit_logits = logits[:, ex, :]
            # print(exit_logits[:10, :], labels[:10, :])
            loss += self.alpha[ex] * gamma[ex] * self.bce(exit_logits, labels) #F.cross_entropy(exit_logits, labels)

        return loss


def try_train(pooled=False, earlyexit=False):
    from util.traineval import train
    from uncertainty import evaluate_model_on_tests, evaluate
    # from audio_res_net_18 import ResNet18EarlyExit, ResNet18
    device= "cuda" if torch.cuda.is_available() else "cpu"
    if earlyexit:
        print("early exit")
        model = BaselineEarlyExit()
        # model = ResNet18EarlyExit(exit_after=[1,3,5,7])
        lossfunc = ExitWeightedLoss()
    else:
        print("single exit")
        model = Baseline()
        # model = ResNet18()
        lossfunc = BaselineLoss()
    model.to(device)
    for i in range(1):
        print("client", "pooled" if pooled else i)
        train_dataloader = get_dataloader(center=i, train=True, pooled=pooled)
        test_dataloader = get_dataloader(center=i, train=False, pooled=pooled)
    
        optimizer = Optimizer(model.parameters(), lr=LR)
        for epoch in range(NUM_EPOCHS_CENTRALIZED): # NUM_EPOCHS_CENTRALIZED
            train(model, train_dataloader, optimizer, lossfunc, device)
            # res = evaluate_model_on_tests(model, [train_dataloader], metric, task="binary")
            # print("train acc", res)
            # if epoch % 5 == 4:
            res = evaluate_model_on_tests(model, [train_dataloader], metric, task="binary", earlyexit=earlyexit)
            print("train acc", res["client_test_0"])
            res = evaluate_model_on_tests(model, [test_dataloader], metric, task="binary", earlyexit=earlyexit)
            print("test acc", res["client_test_0"])
            # break
        
        dict_cindex, y_true, y_pred, variance, entropy = evaluate_model_on_tests(model,  [test_dataloader], metric, MCDO=False, return_pred=True, earlyexit=earlyexit)
        evaluate(dict_cindex, y_true, y_pred, variance, entropy, uncertainty=False, id="pcg_centralized_earlyexit" + str(earlyexit), task="binary")



#=======get 101*99 spectrogram features
def get_spectrogram(segment, audio):
    segment=segment/ (np.sqrt(np.sum(audio*audio)/len(audio)))
    f, t, Sxx = signal.spectrogram(segment, fs=2000, nperseg=200, noverlap=100,mode='magnitude')
    return Sxx


def process_spectogram_nilanon():
    segment_dur_secs = 10
    sr = 2000
    segment_length = sr * segment_dur_secs
    
    # data_dir = "datautil/PCG/"
    # data_dir = "~/rds/hpc-work/data_util/PCG/" // don't work when running
    data_dir = "/home/yz798/rds/hpc-work/data_util/PCG/"
    datasets = ["training-" + idx + "/" for idx in "abcdef"]
    for dataset in datasets[:]:
        contents = []
        input_directory = data_dir + dataset
        file_list = glob.glob(input_directory + "*.wav", recursive=True)
        num_files = len(file_list)
        print(num_files)
        X, Y = [], []
        for filename in file_list:
            sfile_name = filename.split('/')[-1].split('.')[0]
            header_file_name = filename[:-3] + "hea"
            file_label = DataReader.read_label(header_file_name)
            y, _ = librosa.load(filename, sr=sr)
            # print(librosa.get_duration(y=y, sr=sr))
            num_sections = int(np.floor(len(y) / segment_length))
            # print(num_sections)
            for i in range(num_sections):
                t = y[i * segment_length: (i + 1) * segment_length]
                image_file_name = sfile_name + "_" + str(i)
                # x = get_spectrogram(t, y)
                x = get_spectrogram(y, y)
                x = x.reshape(1,101,99)
                X.append(x)
                Y.append(file_label)
        X = np.array(X)
        Y = np.array(Y)
        print(X.shape)
        print(Y.shape)
        # np.save(input_directory + "X.npy", X)
        # np.save(input_directory + "Y.npy", Y)
        np.save(input_directory + "X_noseg.npy", X)
        np.save(input_directory + "Y_noseg.npy", Y)

        

def try_nilanon():
    process_spectogram_nilanon()
    
    

if __name__ == '__main__':
    print("running PCG dataset debugging")
    # tmp_har()
    # header_file_name = "/home/yz798/L46/PersonalizedFL/datautil/PCG/training-a/a0007.hea"
    # label = DataReader.read_label(header_file_name)
    # print(label)
    # print("=" * 48)
    # for p in [True, False][1:]:
    #     for e in [True, False]:
    #         try_train(p, e)
    # print(get_nb_max_rounds(20))
    for i in NUM_CLIENTS:
        get_dataloader(i)
    # try_train(pooled=False, earlyexit=False)
    # try_train(pooled=True, earlyexit=True)
    # try_nilanon()
    # tmp_pcg()
    # process_spectogram()
    