import torch
import torch.nn as nn
import argparse
import pandas as pd
import numpy as np
from features import \
    num_atom_features, \
    num_bond_features
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys
from networkE import dockingProtocol
from util import *
import time
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler, MinMaxScaler

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('-dropout','--df',required=True)
parser.add_argument('-learn_rate','--lr',required=True)
parser.add_argument('-os','--os',required=True)
parser.add_argument('-bs', '--batch_size', required=True)
parser.add_argument('-fplen', '--fplength', required=True)
parser.add_argument('-mnum', '--model_number', required=True)
parser.add_argument('-wd', '--weight_decay', required=True)
parser.add_argument('-ba', '--bin_array', required=True)
parser.add_argument('-hl', '--homolumo', required=True)

cmdlArgs = parser.parse_args()
df=float(cmdlArgs.df)
lr=float(cmdlArgs.lr)
oss=int(cmdlArgs.os)
wd=float(cmdlArgs.weight_decay)
bs=int(cmdlArgs.batch_size)
fplCmd = int(cmdlArgs.fplength)
mn = cmdlArgs.model_number
ba = cmdlArgs.bin_array
ba = [fplCmd] + list(map(lambda x: int(fplCmd / x), list(map(int, ba.split(","))))) + [1]
homolumo = cmdlArgs.homolumo
print(ba)

print(f'interop threads: {torch.get_num_interop_threads()}, intraop threads: {torch.get_num_threads()}')

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.closs = 0
        self.ccounter = 0

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif np.abs([self.min_validation_loss - validation_loss])[0] <= self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def early_cstop(self, train_loss):
        if train_loss == self.closs:
            self.ccounter += 1
        else:
            self.closs = train_loss
            self.ccounter = 0
        if self.ccounter == 200:
            return True
        return False

res_path = f'./../model{mn}'
print(res_path)
try:
    os.mkdir(f'./../model{mn}')
except:
    print("error in creating res dir")

hl = homolumo
# 70-15-15 split

data_path = f'../../data/hldata.csv'
allData = pd.read_csv(data_path)

trainData, valData, testData = np.split(allData.sample(frac=0.3), 
                                        [int(.70*len(allData)*0.3), int(.85*len(allData)*0.3)])

print(f'merged df shapes: {trainData.shape}, {valData.shape}, {testData.shape}')
print(trainData.head())

# ad = None
# with open("../../data/ammoniums.csv", 'r') as f:
#     d = [x.strip().split(",") for x in f.readlines()[1:]]
#     # hardness, en
#     derived = np.array([list(map(float, x[-2:])) for x in d])
#     l, h = derived[:, 0] - derived[:, 1], -(derived[:, 0] + derived[:, 1])
#     print(h, l)
#     ad = [d[i][:2] + [h[i]] + [l[i]] for i in range(len(d))]

# xTrain = np.concatenate((trainData[['file', 'smiles']].values.tolist(), [x[:2] for x in ad]))
# yTrain = np.concatenate((trainData[hl].values, [x[2 if hl == "homo" else 3] for x in ad]))
# xTest = np.concatenate((testData[['file', 'smiles']].values.tolist(), [x[:2] for x in ad]))
# yTest = np.concatenate((testData[hl].values, [x[2 if hl == "homo" else 3] for x in ad]))
# xValid = np.concatenate((valData[['file', 'smiles']].values.tolist(), [x[:2] for x in ad]))
# yValid = np.concatenate((valData[hl].values, [x[2 if hl == "homo" else 3] for x in ad]))

xTrain = trainData[['file', 'smiles']].values.tolist()
yTrain = trainData[hl].values
xTest = testData[['file', 'smiles']].values.tolist()
yTest = testData[hl].values
xValid = valData[['file', 'smiles']].values.tolist()
yValid = valData[hl].values

with open(f'../model{mn}/testset.txt', 'w') as f:
    f.write("id,smile,hl\n")
    for (id_s, smile), hl in zip(xTest, yTest):
        f.write(f'{id_s},{smile},{hl}\n')

print(f"xTrain: {xTrain[:5]}")
print(f"yTrain: {yTrain[:5]}")

scaler = StandardScaler()
yTrain = yTrain.reshape(-1, 1)
yTrain = scaler.fit_transform(yTrain).T[0].tolist()  
yTest = scaler.transform(yTest.reshape(-1, 1)).T[0].tolist()  # reuse scaling from train data to avoid data leakage
yValid = scaler.transform(yValid.reshape(-1, 1)).T[0].tolist()

trainds = dockingDataset(train=xTrain, 
                        labels=yTrain,
                        name='train')
traindl = DataLoader(trainds, batch_size=bs, shuffle=True)
testds = dockingDataset(train=xTest,
                        labels=yTest,
                        name='test')
testdl = DataLoader(testds, batch_size=bs, shuffle=True)
validds = dockingDataset(train=xValid,
                         labels=yValid,
                         name='valid')
validdl = DataLoader(validds, batch_size=bs, shuffle=True)


fpl = fplCmd 
hiddenfeats = [fpl] * 4  # conv layers, of same size as fingeprint (so can map activations to features)
layers = [num_atom_features()] + hiddenfeats
modelParams = {
    "fpl": fpl,
    "batchsize": bs,
    "conv": {
        "layers": layers,
        "activations": False
    },
    "ann": {
        "layers": layers,
        "ba": ba,
        "dropout": df
    }
}
print(f'layers: {layers}, through-shape: {list(zip(layers[:-1], layers[1:]))}')

model = dockingProtocol(modelParams).to(device=device)
print(model)
# print("inital grad check")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
totalParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'total trainable params: {totalParams}')
lossFn = nn.MSELoss() # gives 'mean' val by default
# adam, lr=0.01, weight_decay=0.001, prop=0.2, dropout=0.2
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
model.load_state_dict(torch.load(f'../basisModel{mn}.pth'), strict=False)
lendl = len(trainds)
num_batches = len(traindl)
print("Num batches:", num_batches)
bestVLoss = 100000000
lastEpoch = False
epochs = 200  # 200 initially 
earlyStop = EarlyStopper(patience=20, min_delta=0.01)
converged_at = 0
trainLoss, validLoss = [], []
trainR, validR = [], []
for epoch in range(1, epochs + 1):
    print(f'\nEpoch {epoch}\n------------------------------------------------')
    
    stime = time.time()
    model.train()
    runningLoss, r_squared, r_list = 0, 0, []

    for batch, (a, b, e, (y, zidTr)) in enumerate(traindl):
        at, bo, ed, scaled_Y = a.to(device), b.to(device), e.to(device), y.to(device)

        scaled_preds = model((at, bo, ed))
        loss = lossFn(scaled_preds, scaled_Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        runningLoss += scaled_preds.shape[0] * loss.item()
 
        preds = scaler.inverse_transform(scaled_preds.detach().cpu().numpy().reshape(-1, 1)).T[0].tolist()
        Y = scaler.inverse_transform(scaled_Y.detach().cpu().numpy().reshape(-1, 1)).T[0].tolist()
        
        if batch == 0:
            print(f"Pred: {preds[:3]} vs True: {Y[:3]}")

        _,_,r_value,_,_ = linregress(preds, Y)
        r_list.append(r_value ** 2)

        cStop = earlyStop.early_cstop(loss.item())
        if cStop: break
   
        if batch % (np.ceil(lendl / bs / 10)) == 0:
            lossDisplay, currentDisplay = loss.item(), (batch + 1)
            print(f'loss: {lossDisplay:>7f} [{((batch + 1) * len(a)):>5d}/{lendl:>5d}]')

    trainLoss.append(runningLoss/lendl)
    r_squared = 0
    if len(r_list) != 0:
        r_squared = sum(r_list)/len(r_list)
    trainR.append(r_squared)
    if cStop: break
    print(f'Time to complete epoch: {time.time() - stime}')
    print(f'\nTraining Epoch {epoch} Results:\nloss: {runningLoss/lendl:>8f}, R^2: {r_squared:>8f}\n------------------------------------------------')
    
    size = len(validdl.dataset)
    num_batches = len(validdl)
    model.eval()
    valid_loss = 0
    runningLoss, r_squared, r_list = 0, 0, []
    with torch.no_grad():
        for (a, b, e, (scaled_y, zidValid)) in validdl:
            scaled_preds = model((a, b, e))
            valid_loss += lossFn(scaled_preds.to(device), scaled_y.to(device)).item()

            preds = scaler.inverse_transform(scaled_preds.detach().cpu().numpy().reshape(-1, 1)).T[0].tolist()
            y = scaler.inverse_transform(scaled_y.detach().cpu().numpy().reshape(-1, 1)).T[0].tolist()
            _, _, r_value, _, _ = linregress(preds, y)
            r_list.append(r_value ** 2)
    valid_loss /= num_batches
    if len(r_list) != 0:
        r_squared = sum(r_list)/len(r_list)
    validLoss.append(valid_loss)
    validR.append(r_squared)
    print(f'\nValidation Results:\nLoss: {valid_loss:>4f}, R^2: {r_squared:>2f}%\n------------------------------------------------')
    
    if valid_loss < bestVLoss:
        bestVLoss = valid_loss
        model_path = f'../model{mn}/checkpoint.pth'
        print(f"Saved current as new best.")
        model.save(modelParams, hl, model_path, scaler)

    if earlyStop.early_stop(valid_loss):
        print(f'validation loss converged to ~{valid_loss}')
        converged_at = epoch
        break

if cStop: 
    print(f'training loss converged erroneously')
    sys.exit(0)

if converged_at != 0:
    epochR = range(1, converged_at + 1)
else:
    epochR = range(1, epoch + 1)
plt.plot(epochR, trainLoss, label='Training Loss', linestyle='-', color='lightgreen')
plt.plot(epochR, validLoss, label='Validation Loss', linestyle='-', color='darkgreen')
plt.plot(epochR, trainR, label='Training R^2', linestyle='--', color='lightblue')
plt.plot(epochR, validR, label='Validation R^2', linestyle='--', color='darkblue')

plt.title('Training and Validation Loss/R^2')
plt.xlabel('Epochs')
plt.ylabel('Loss / R^2')
 
plt.legend(loc='best')

cepoch = epochs if converged_at == 0 else converged_at
plt.xticks(np.arange(0, cepoch+1, cepoch//10 if cepoch > 15 else cepoch))
 
plt.legend(loc='best')
plt.savefig(f'{res_path}/rloss.png')
plt.close()
with open(f'{res_path}/loss.txt', 'w+') as f:
    f.write('train loss, validation loss\n')
    f.write(f'{",".join([str(x) for x in trainLoss])}')
    f.write(f'{",".join([str(x) for x in validLoss])}')