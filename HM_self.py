import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from function import get_device, plot_learning_curve, plot_pred

tr_path = 'covid.train.csv'  # path to training data
tt_path = 'covid.test.csv'  # path to testing data


class Mydataset(Dataset):
    def __init__(self, path, mode='train'):
        self.mode = mode
        with open(path, 'r') as file:  # read csv file
            data = list(csv.reader(file))
            data = np.array(data[1:])[:, 1:].astype(float)  # we dont want first coloum

        # maybe we can do some better feature extract here.....?

        feats = list(range(93))
        if mode == "test":
            feature = data[:, feats]
            self.feature = torch.FloatTensor(feature)
        else:
            feature = data[:, feats]
            label = data[:, -1]
            if mode == 'train':
                indicate = [i for i in range(len(feature)) if i % 10 != 1]
            elif mode == 'dev':
                indicate = [i for i in range(len(feature)) if i % 10 == 1]

            self.feature = torch.FloatTensor(feature[indicate])
            self.label = torch.FloatTensor(label[indicate])

        #  normalization
        self.feature[:, 40:] = \
            (self.feature[:, 40:] - self.feature[:, 40:].mean(dim=0, keepdim=True)) \
            / self.feature[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.feature.shape[1]

    def __getitem__(self, item):
        # we should set self.mode at upper so that we can use mode inside this function
        if self.mode == 'test':
            return self.feature[item]
        else:
            return self.feature[item], self.label[item]

    def __len__(self):
        return len(self.feature)


def prep_dataloader(path, mode, batch_size=0, n_job=0):
    dataset = Mydataset(path, mode)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_job, pin_memory=True, shuffle=(mode == 'train'))
    return loader


class Module(nn.Module):
    def __init__(self, input_dim):
        super(Module, self).__init__()
        # we can define more complicate layer here...
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # loss function
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, x):
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        return self.loss(pred, target)


# train, dev, test modle def
def dev(dataset, model, device):
    model.eval()
    total_loss = 0
    for x, y in dataset:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
        total_loss += mse_loss.detach().cpu().item() * len(x)
    total_loss = total_loss / len(dataset.dataset)
    return total_loss


def train(tr_set, dev_set, model, config, device):

    n_epochs = config['n_epochs']

    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])

    mis_mse = 1000

    loss_record = {'train': [], 'dev': []}

    early_stop_cnt = 0

    epoch = 0
    while epoch < n_epochs:
        model.train()
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
            mse_loss.backward()
            optimizer.step()
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # dev
        dev_mse = dev(dev_set, model, device)
        if dev_mse < mis_mse:
            mis_mse = dev_mse
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            break
    print("finished training...")
    return mis_mse, loss_record


def test(test_set, model, config, device):
    model.eval()
    preds = []
    for x in test_set:
        x = x.to(device)
        with torch.no_grad:
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


device = get_device()
os.makedirs('models', exist_ok=True)

config = {
    'n_epochs': 3000,
    'batch_size': 270,
    'optimizer': 'SGD',
    'optim_hparas':{
        'lr': 0.001,
        'momentum': 0.9
    },
    'early_stop': 200,
    'save_path': 'models/model.pth'
}

tr_set = prep_dataloader(tr_path, 'train', config['batch_size'])
dev_set = prep_dataloader(tr_path, 'dev', config['batch_size'])
test_set = prep_dataloader(tt_path, 'test', config['batch_size'])

model = Module(tr_set.dataset.dim).to(device)

model_loss, model_loss_record = train(tr_set, dev_set, model, config, device)

print(model_loss)
print(model_loss_record)
