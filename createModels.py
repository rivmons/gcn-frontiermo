import os
import argparse
import pandas as pd
from networkE import dockingProtocol
from features import num_atom_features
from torch import save
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-hl', '--homolumo', required=True)

ioargs = parser.parse_args()
homolumo = ioargs.homolumo
dir_name = homolumo

dropout = [0.0]
learn_rate = [0.001, 0.0001, 0.0003]
weight_decay = [0.0001, 0]
bs = [128, 256]
fpl = [32, 64, 128]
ba = np.array([[2, 4], [2, 4, 8], [2, 8]], dtype=object) # through shape of ANN (divide by fingerprint size)

# preset hps array if want to use
models_hps = [
    [0, 0.001, 0.0001, 25, 64, 32],
    [0, 0.001, 0, 25, 128, 32],
    [0, 0.0001, 0.0001, 25, 256, 64],
    [0, 0.001, 0.0001, 25, 128, 32],
    [0, 0.001, 0.0001, 25, 256, 128]
]

hps = []
# number of models; random sample; can change
for i in range(10):
    hps.append([
        25,
        np.random.choice(bs),
        0,
        np.random.choice(learn_rate),
        np.random.choice(weight_decay),
        np.random.choice(fpl),
        np.random.choice(ba)
    ])

print(f'num models: {len(hps)}')                               

try:
    os.mkdir(f'./{dir_name}')
except:
    pass

try:
    os.mkdir(f'./{dir_name}/trainingJobs')
except:
    pass

try:
    os.mkdir(f'./{dir_name}/logs')
except:
    pass

# with open(f'./{dir_name}/hpResults.csv', 'w+') as f:
#     f.write(f'model number,oversampled size,batch size,learning rate,dropout rate,gfe threshold,fingerprint length,validation auc,validation prauc,validation precision,validation recall,validation f1,validation hits,test auc,test prauc,test precision,test recall,test f1,test hits\n')
    

for f in os.listdir(f'./{dir_name}/trainingJobs/'):
    os.remove(os.path.join(f'./{dir_name}/trainingJobs', f))
for f in os.listdir(f'./{dir_name}/logs/'):
    os.remove(os.path.join(f'./{dir_name}/logs', f))
    
for i in range(len(hps)):
    with open(f'./{dir_name}/trainingJobs/train{i + 1}.sh', 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write(f'cd ./{dir_name}/trainingJobs\n')
        # f.write('module load python-libs/3.0\n')
 
        o,batch,do,lr,wd,fpl,ba = hps[i]
        f.write('python '+'../../train.py'+' '+'-dropout'+' '+str(do)+' '+'-learn_rate'+' '+str(lr)+' '+'-os'+' '+str(o)+' '+'-bs'+' '+str(batch)+' '+' '+'-fplen '+str(fpl)+' '+'-wd '+str(wd)+' '+'-mnum '+str(i+1)+' '+'-ba '+','.join(list(map(str, ba)))+' '+'-hl '+str(homolumo)+'\n')


# need to update when updating model params
for i, m in enumerate(hps):
    fpl = int(m[-2]) 
    ba = m[-1]
    print(ba,  [fpl] + list(map(lambda x: int(fpl / x), ba)) + [1])
    hiddenfeats = [fpl] * 4  # conv layers, of same size as fingeprint (so can map activations to features)
    layers = [num_atom_features()] + hiddenfeats
    modelParams = {
        "fpl": fpl,
        "activation": 'regression',
        "conv": {
            "layers": layers,
            "activations": False
        },
        "ann": {
            "layers": layers,
            "ba": [fpl] + list(map(lambda x: int(fpl / x), ba)) + [1],
            "dropout": 0.1 #arbitrary
        }
    }
    model = dockingProtocol(modelParams)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model trainable params: {pytorch_total_params}')
    save(model.state_dict(), f'./{dir_name}/basisModel{i+1}.pth')
