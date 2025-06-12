import os
import time
import datetime
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset import * 
from model import * 
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, roc_auc_score, r2_score


def load_best_result(model):
    best_ckpt_path = "E:/LingjingZhang/dnatopo/brics_a/a_input_pro4.0/seed_3407/checkpoints/best_ckpt.pth"
    ckpt = torch.load(best_ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model'])

    return model

model = HiGNN(in_channels=46,
              hidden_channels=256,
              out_channels=1,
              edge_dim=10,
              num_layers=4,
              dropout=0.1,
              slices=4,
              f_att=True,
              r=4,
              brics=True,
              cl=False)

model = load_best_result(model)

path = "./data/a_input_pro/"
dataset = 'a_input_plus_random'
task_type = 'regression'
tasks = 'pIC50(M)'
aic50 = MolDataset(root=path, dataset=dataset, task_type=task_type, tasks=tasks)

seed = 3407
random = Random(seed)
indices_1 = list(range(len(aic50)))
random.seed(seed)

train_size = int(0.75 * len(aic50))
val_size = int(0.05 * len(aic50))
test_size = len(aic50) - train_size - val_size

indices_2 = []
for i in indices_1[:(train_size + val_size)]:
    indices_2.append(i)
random.shuffle(indices_2) #ZLJ
trn_id, val_id, test_id = indices_2[:train_size], \
                          indices_2[train_size:(train_size + val_size)], \
                          indices_1[(train_size + val_size):]

aic50 = aic50[test_id]


loader = DataLoader(aic50, batch_size=154)

iter_ = iter(loader)
batch = next(iter_)

model.eval()  # 关闭dropout
output = model(batch)

pred = output[0].detach().numpy()
#print(pred[:10])
#print(len(pred))

att = output[1][0].detach().numpy()


cross = output[1][1].detach().numpy()
#print(cross)

idx = cross[1]


i = 6

df = pd.read_csv("./data/a_input_pro/raw/a_input_plus_random.csv")
df_test = df.iloc[test_id]

num = np.where(idx==i)[0]

#print(cross[:,num])

#print(att[num])

smiles = df_test['smiles'].iloc[i]
mol = Chem.MolFromSmiles(smiles)
print(smiles)
"""
drawing
"""

from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem import Draw


results = np.array(sorted(list(FindBRICSBonds(mol))), dtype=np.long)
bond_to_break = results[:, 0, :]
bond_to_break = bond_to_break.tolist()
with Chem.RWMol(mol) as rwmol:
    for i in bond_to_break:
        rwmol.RemoveBond(*i)
rwmol = rwmol.GetMol()


#results

cluster_idx = []
Chem.rdmolops.GetMolFrags(rwmol, asMols=True, sanitizeFrags=False, frags=cluster_idx)
#cluster_idx

atoms = mol.GetAtoms()

hit_ats = list(range(0, len(atoms)))

weight_atom = att[num][cluster_idx]
weight_atom = (weight_atom - weight_atom.min())/(weight_atom.max() - weight_atom.min())
#weight_atom

hit_bonds = []
weight_bond = []
for bond in mol.GetBonds():
    aid1 = hit_ats[bond.GetBeginAtomIdx()]
    aid2 = hit_ats[bond.GetEndAtomIdx()]
    hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
    
    if weight_atom[aid1] == weight_atom[aid2]:
        weight_bond.append(weight_atom[aid1])
    else:
        weight_bond.append(0)
#weight_bond

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

norm = matplotlib.colors.Normalize(vmin=0,vmax=1.5)
cmap = cm.get_cmap('Oranges')
plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)

atom_cols = {}
for at in hit_ats:
    atom_cols[at] = plt_colors.to_rgba(float(weight_atom[at]))
    
bond_cols = {}
for bd in hit_bonds:
    bond_cols[bd] = plt_colors.to_rgba(float(weight_bond[bd]))

d = Draw.MolDraw2DCairo(1000, 1000)
#d.SetDPI(1000)
Draw.PrepareAndDrawMolecule(d, mol, highlightAtoms=hit_ats,
                                   highlightBonds=hit_bonds,
                                  highlightAtomColors=atom_cols,
                                  highlightBondColors=bond_cols)

d.FinishDrawing()
# 保存加权分子图的 SVG 文件
with open('./result/weighted_molecule_6.png', 'wb') as f:
    f.write(d.GetDrawingText())
