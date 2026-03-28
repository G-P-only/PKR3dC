# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
import pandas as pd
from matplotlib.colors import ListedColormap
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from PKR3dC.preprocessing_aff import create_affinity_graph, read_data
rdDepictor.SetPreferCoordGen(True)
from IPython.display import SVG
import cairosvg
import cv2
import matplotlib.cm as cm
from tqdm import tqdm
from PKR3dC.model.Pkr3dc import PKR3dC
from PKR3dC.tools.vis import *

class GradCAM_GVP():
    def __init__(self, model, module):
        self.model = model
        module.register_forward_hook(self.save_hook)
        self.target_feat = None

    def save_hook(self, md, fin, fout):
        # self.target_feat = fout
        self.target_feat = fout[0]

    def __call__(self, data):
        self.model.eval()

        output = self.model(data).view(-1)
        grad = torch.autograd.grad(output, self.target_feat,retain_graph=True)[0]
        channel_weight = torch.mean(grad, dim=0, keepdim=True)
        channel_weight = normalize(channel_weight[0])
        weighted_feat = self.target_feat * channel_weight
        cam = torch.sum(weighted_feat, dim=-1).detach().cpu().numpy()
        cam = normalize(cam)

        return output.detach().cpu().numpy(), cam


def clourMol(mol, highlightAtoms_p=None, highlightAtomColors_p=None, highlightBonds_p=None, highlightBondColors_p=None,
             sz=[400, 400], radii=None):
    d2d = rdMolDraw2D.MolDraw2DSVG(sz[0], sz[1])
    op = d2d.drawOptions()
    op.dotsPerAngstrom = 40
    op.useBWAtomPalette()
    mc = rdMolDraw2D.PrepareMolForDrawing(mol)
    d2d.DrawMolecule(mc, legend='', highlightAtoms=highlightAtoms_p, highlightAtomColors=highlightAtomColors_p,
                     highlightBonds=highlightBonds_p, highlightBondColors=highlightBondColors_p,
                     highlightAtomRadii=radii)
    d2d.FinishDrawing()
    svg = SVG(d2d.GetDrawingText())
    res = cairosvg.svg2png(svg.data, dpi=600, output_width=2400, output_height=2400)
    nparr = np.frombuffer(res, dtype=np.uint8)
    segment_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return segment_data

class GNNDataset(InMemoryDataset):

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['phy.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_fold_0.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def process(self):
        pass

def main():
    device = torch.device('cuda:0')
    test_df = pd.read_csv(os.path.join('./', 'raw', 'phy.csv'))
    test_set = GNNDataset('./', train=False)
    ds='davis'
    # ds='kiba'
    affinity = read_data(ds)
    affinity_graph = create_affinity_graph(affinity, ds)
    model = PKR3dC(affinity_graph=affinity_graph, dataset=ds).to(device)
    try:
        load_model_dict(model, '../data/best_model.pt')
    except:
        model_dict = torch.load('../data/best_model.pt')
        for key, val in model_dict.copy().items():
            if 'lin_l' in key:
                new_key = key.replace('lin_l', 'lin_rel')
                model_dict[new_key] = model_dict.pop(key)
            elif 'lin_r' in key:
                new_key = key.replace('lin_r', 'lin_root')
                model_dict[new_key] = model_dict.pop(key)
        model.load_state_dict(model_dict)

    gradcam = GradCAM_GVP(model, module=model.protein_model.layers[2].conv)

    bottom = cm.get_cmap('Blues_r', 256)
    top = cm.get_cmap('Oranges', 256)
    newcolors = np.vstack([bottom(np.linspace(0.35, 0.85, 128)), top(np.linspace(0.15, 0.65, 128))])
    newcmp = ListedColormap(newcolors, name='OrangeBlue')
    test_df.columns = test_df.columns.str.lower()
    smile_list = list(test_df['smiles'].unique())

    progress_bar = tqdm(total=len(smile_list))

    for idx in range(len(test_set)):
        smile = test_df.iloc[idx]['smiles']

        if len(smile_list) == 0:
            break
        if smile in smile_list:
            smile_list.remove(smile)
        else:
            continue

        data = Batch.from_data_list([test_set[idx]])
        data = data.to(device)
        _, atom_att = gradcam(data)

        mol = Chem.MolFromSmiles(smile)
        atom_color = dict([(idx, newcmp(atom_att[idx])[:3]) for idx in range(len(atom_att))])
        radii = dict([(idx, 0.2) for idx in range(len(atom_att))])
        img = clourMol(mol, highlightAtoms_p=range(len(atom_att)), highlightAtomColors_p=atom_color, radii=radii)

        cv2.imwrite(os.path.join('./result', f'{ds}{idx}.png'), img)

        progress_bar.update(1)


if __name__ == '__main__':
    main()

