import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import torch
from networkE import dockingProtocol, dockingDataset, nfpDocking, Ensemble
from torch.utils.data import DataLoader
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from features import getAtomFeatures, num_atom_features
import os
import io
from PIL import Image
import glob
import statistics
import argparse
from util import buildFeats
import pandas as pd
import sys
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def get_metrics(e, r_e):
    return root_mean_squared_error(r_e, e)

def get_all_metrics(e, r_e):
    return (root_mean_squared_error(r_e,e), mean_squared_error(r_e, e), mean_absolute_error(r_e,e), smape(r_e, e))

def get_preds(modelp, modeld, scaler, xtest):
    model = dockingProtocol(modelp).to(device=device)
    model.load_state_dict(modeld['model_state_dict'], strict=False)
    model.eval()
    testds = dockingDataset(train=xtest,
                            labels=[0] * len(xtest),
                            name='test')
    testdl = DataLoader(testds, batch_size=512, shuffle=False)

    mol = 0
    preds_s = None
    for (a, b, e, (y, zid)) in testdl:
        preds = model((a, b, e))
        preds = scaler.inverse_transform(preds.detach().cpu().numpy().reshape(-1, 1)).T[0].tolist()
        if preds_s == None:
            preds_s = preds
        else: preds_s = torch.concatenate(preds, preds_s)
    model = None
    return preds_s

def get_ens_preds(path, xtest):
    modelp = None
    modeld = torch.load(f'{path}/checkpoint.pth', map_location=torch.device(device))
    scaler = modeld["scaler"]

    model = None
    model = Ensemble(modeld["params"]["num_models"], *(modeld["params"]["models"])).to(device=device)
    model.load_state_dict(modeld['model_state_dict'], strict=False)
    model.eval()

    testds = dockingDataset(train=xtest,
                            labels=np.zeros((len(xtest))),
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
    return preds_s

# ATOM ACTIVATIONS/SUBSTRUCTURES

def remove_duplicates(values, key_lambda):
    output = []
    seen = set()
    for value in values:
        # If value has not been encountered yet, add it to both list and set.
        cur_key = key_lambda(value)
        if cur_key not in seen:
            output.append(value)
            seen.add(cur_key)
    return output

def get_substructure(atom, radius):
        # Recursive function to get indices of all atoms in a certain radius.
        if radius == 0:
            return set([atom.GetIdx()])
        else:
            cur_set = set([atom.GetIdx()])
            for neighbor_atom in atom.GetNeighbors():
                cur_set.update(get_substructure(neighbor_atom, radius - 1))
            return cur_set
        
def draw(molecule, substructure_idxs, fpix, figix):

    bonds = set()
    for idx in substructure_idxs:
        for idx_1 in substructure_idxs:
            if molecule.GetBondBetweenAtoms(idx, idx_1): bonds.add(molecule.GetBondBetweenAtoms(idx, idx_1).GetIdx())

    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(350,300)
    drawer.drawOptions().fillHighlights=True
    drawer.drawOptions().setHighlightColour((1.0, 0.0, 0.0, 0.2))
    drawer.drawOptions().highlightBondWidthMultiplier=10
    drawer.drawOptions().useBWAtomPalette()
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule, highlightAtoms=substructure_idxs, highlightBonds=list(bonds))
    bio = io.BytesIO(drawer.GetDrawingText())
    im = Image.open(bio)
    im = im.save(f"./substructure_activations/fp{fpix}_{figix}.png")

def plot(activations):
    
    # shape (6, 64, 70, 32)
    # rep (degrees, batch size, atoms with max, hf or fpl?)
    # need to range over mols
    try: os.rmdir("./substructure_activations")
    except: pass
    try: os.mkdir("./substructure_activations")
    except: pass

    of = open('./substructure_activations/activations.txt', 'w')
    of.write('fingerprint_index,nth_best_activation,fingeprint_index_activation,most_active_atom_ix,most_active_mol_ix,radius,file,smile,homo,lumo\n')

    for fpix in range(modelp["fpl"]):
        fpix_list = []
        for mol_ix in range(len(xtest)):
            # if activations_zid[mol_ix] not in xhits: continue
            for rad in range(len(modelp["conv"]["layers"])):
                fp_activations = activations[rad][mol_ix, :, fpix]
                fp_activations = fp_activations[fp_activations != 0]
                fpix_list += [(fp_activation, atom_ix, mol_ix, rad) for atom_ix, fp_activation in enumerate(fp_activations)]
       
        unique_list = remove_duplicates(fpix_list, key_lambda=lambda x: x[0])
        fpix_list = sorted(unique_list, key=lambda x: -x[0])

        for fig_ix in range(10):
            # Find the most-activating atoms for this fingerprint index, across all molecules and depths.
            activation, most_active_atom_ix, most_active_mol_ix, ra = fpix_list[fig_ix]
            print(activation, most_active_atom_ix, most_active_mol_ix, ra, activations_zid[most_active_mol_ix])
            of.write(f'{fpix},{fig_ix},{activation},{most_active_atom_ix},{most_active_mol_ix},{ra},{activations_zid[most_active_mol_ix]},{smilesdict[activations_zid[most_active_mol_ix]]}.{zinc_gfe_d[activations_zid[most_active_mol_ix]][1]},{zinc_gfe_d[activations_zid[most_active_mol_ix]][2]}\n')
            ma_smile = smilesdict[activations_zid[most_active_mol_ix]]
            molecule = Chem.MolFromSmiles(ma_smile)
            ma_atom = molecule.GetAtoms()[most_active_atom_ix]
            substructure_idxs = get_substructure(ma_atom, ra)

            draw(molecule, list(substructure_idxs), fpix, fig_ix)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument('-hl', '--homolumo', required=True)
    parser.add_argument('-mn', '--model', required=True)

    ioargs = parser.parse_args()
    path = ioargs.homolumo
    mn = ioargs.model

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    smilesdict = None
    xtest, ytest = [], []
    actives_d, decoys_d = {}, {}
    xhits, xnonhits = [], []
    cf = None
    zinc_gfe_d = {}
    modelp = None
    modeld = None

    modelp = None
    modeld = torch.load(f'{path}/model{mn}/checkpoint.pth', map_location=torch.device(device))
    modelp = modeld['params']
    scaler = modeld['scaler']

    testf = open(f'{path}/model{mn}/testset.txt', 'r')
    testd = [i.strip().split(",") for i in testf.readlines()[1:]]

    smilesdict = {
        i[0]: i[1]
        for i in testd
    }
    data_path = f'./data/hldata.csv'
    allData = pd.read_csv(data_path)
    zinc_gfe_d = allData.set_index('file').T.to_dict('list')
    xtest = [[i[0], i[1]] for i in testd]
    ytest = [float(i[2]) for i in testd]
    print(xtest[:2], ytest[:2])

    # [zid, smile], 0/1
    modelp["conv"]["activations"] = True
    model = dockingProtocol(modelp).to(device=device)
    model.load_state_dict(modeld['model_state_dict'], strict=False)
    model.eval()
    print(len(xtest), len(ytest))
    testds = dockingDataset(train=xtest,
                            labels=ytest,
                            name='test')
    testdl = DataLoader(testds, batch_size=512, shuffle=False)
    print('done building tensors')

    activations = np.empty((len(modelp["conv"]["layers"]), len(xtest), 100, modelp["fpl"]))
    activations_zid = []
    mol = 0
    preds_s = None
    for (a, b, e, (y, zid)) in testdl:
        preds, bact = model((a, b, e))
        preds = scaler.inverse_transform(preds.detach().cpu().numpy().reshape(-1, 1)).T[0].tolist()
        preds_s = preds
        activations_zid += zid
        # print(a.shape, b.shape, e.shape)
        for i, x in enumerate(bact):
            # print(mol, mol+bact[0].shape[0], bact[i].shape)
            activations[i][mol:mol+bact[0].shape[0], :, :] = bact[i].detach().cpu().numpy()
        mol += bact[0].shape[0]
    
    plot(activations=activations)