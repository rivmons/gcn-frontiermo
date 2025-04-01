import torch
from networkE import dockingProtocol, dockingDataset, nfpDocking, Ensemble
from torch.utils.data import DataLoader
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def get_preds(e_path, mn, ens=False):
    modelp = None
    modeld = None
    if ens:
        modeld = torch.load(f'{e_path}/checkpoint.pth', map_location=torch.device(device))
    else:
        modeld = torch.load(f'{e_path}/model{mn}/checkpoint.pth', map_location=torch.device(device))
    scaler = modeld["scaler"]
    
    xtest, ytest = None, None
    if not ens:
        data = open(f'./{e_path}/model{mn}/testset.txt', 'r').readlines()[1:]
    else:
        data = open(f'./{e_path}/testset.txt', 'r').readlines()[1:]
    test = [x.strip().split(",") for x in data]
    xtest, ytest = [x[:2] for x in test], [float(x[2]) for x in test]

    model = None
    if ens:        
        model = Ensemble(modeld["params"]["num_models"], *(modeld["params"]["models"])).to(device=device)
        model.load_state_dict(modeld['model_state_dict'], strict=False)
    else:
        model = dockingProtocol(modeld["params"]).to(device=device)
    model.eval()

    testds = dockingDataset(train=xtest,
                            labels=ytest,
                            name='test')
    testdl = DataLoader(testds, batch_size=512, shuffle=False)
    print('done building tensors')

    # mol = 0
    preds_s = None
    for (a, b, e, (y, zid)) in testdl:
        preds = model((a, b, e))
        preds = scaler.inverse_transform(preds.detach().cpu().numpy().reshape(-1, 1)).T[0].tolist()
        if preds_s == None:
            preds_s = preds
        else: preds_s += preds

    fn = f"{e_path}_{mn}_preds.csv"
    with open(fn, 'w+') as f:
        f.write(f"drug,smile,calculated_{e_path},predicted_{e_path}\n")
        for i in range(len(xtest)):
            f.write(f'{",".join(xtest[i])},{ytest[i]},{preds_s[i]}\n')
            
    # mae, rmse, r^2
    mae = mean_absolute_error(ytest, preds_s)
    rmse = np.sqrt(mean_squared_error(ytest, preds_s))
    r_2 = r2_score(ytest, preds_s)
    smape_v = smape(np.array(ytest), np.array(preds_s))
    print('-------------------------------------------------')
    print(f'{e_path}-{mn}')
    print(f'mae: {mae}, rmse: {rmse}, r^2: {r_2}, smape: {smape_v}')
    print('-------------------------------------------------')

    plt.scatter(ytest, preds_s)
    plt.savefig(f'{e_path}_{mn}.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--ensemble', required=True)
    parser.add_argument('-m', '--models', required=False)
    parser.add_argument('-d', '--directory', required=True)

    ioargs = parser.parse_args()
    ens = bool(int(ioargs.ensemble))
    dir = ioargs.directory
    if ens:
        print(f'evaluating ensemble at {dir}')
        get_preds(dir, 0, True)
    else:
        models = [int(x) for x in ioargs.models.split(",")]
        print(f'evaluating {", ".join([str(x) for x in models])} in {dir}')
        for i in models:
            get_preds(dir, i, False)