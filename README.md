# PKR3dC: Drug-target Affinity Prediction by Introducing Prior-affinity Knowledge and Residue 3D Coordinate Information
    <p>This work presents PKR3dC, a deep learning approach for drug-target affinity (DTA) prediction that integrates residue 3D coordinate information from PDB files and known affinity values as prior knowledge, addressing the limitation of existing methods that underutilize protein 3D structural information. By using a multi-granularity feature fusion framework, it extracts fine-grained local features from drug molecular graphs, protein 3D structure graphs and sequences through DenseNet, GVP-GNN-transformer and gated CNN, obtains coarse-grained global features from drug-target topological affinity relationships via GCN, and adopts an attention-based module to fuse drug and protein features for better interaction modeling. Evaluated on Davis and KIBA datasets under four experimental settings, PKR3dC outperforms state-of-the-art models, particularly in cold-drug and cold-protein scenarios that simulate real experimental conditions.<p>
## Architecture
![Fig](Fig1.jpg "Model")
## requirements
python==3.7.1<br>
torch==1.9.0+cu111<br>
torch-geometric==2.1.0<br>
torch-scatter==2.0.7<br>
torch-sparse==0.6.10<br>
torch-cluster==1.5.9<br>
torch-spline-conv==1.2.1<br>
networkx==2.6.3 <br>
biopython==1.81<br>
rdkit==2023.3.2<br>
dgllife==0.3.2<br>
deeppurpose==0.1.5<br>
atom3d==0.2.4<br>
freesasa==2.2.1<br>
lifelines==0.27.8<br>
scikit-learn==1.0.2<br>
matplotlib==3.3.4<br>
pandas==1.1.5<br>
numpy==1.21.6<br>
tqdm==4.62.3<br>
plotly==5.18.0<br>
hyperopt==0.2.7

## Datasets download: It includes two public datasets, Davis and Kiba.
https://github.com/G-P-only/PKR3dC/data

## Running
### data processing
`python preprocessing.py`
`python preprocessing_aff.py`
### training(S1,S2,S3,S4)
`python train.py --dataset davis/feicold --save_model --batch_size 64 --lr 1e-4`
`python train_cold.py --dataset davis/(drug、protein、pair)--save_model --batch_size 64 --lr 1e-4 --drug_sim_k 4 --target_sim_k 7`
## visualization
`python visualization.py`
