import os
import random
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import torch.nn.functional as F
import argparse
from metrics import get_cindex, get_rm2
# from model.zhumodel import *
# from model.zhumodelDenseNet import DrugMGR as DrugMGR1
from utils import *
from log.train_logger import TrainLogger
from torch_geometric.data import InMemoryDataset
from wpw.affinitypreprocessing import create_affinity_graph, read_data
# from wpw.mymodels.zhumodelDenseNet import DrugMGR as DrugMGR1
# from wpw.mymodels.zhumodelDenseNetafftoatom import DrugMGR as DrugMGR1
# from wpw.mymodels.zhumodelDenseNetgvpgnntognn import DrugMGR as DrugMGR1
from wpw.mymodels.zhumodelDenseNetnoDensenet import DrugMGR as DrugMGR1

class GNNDataset(InMemoryDataset):
    def __init__(self, root, index, types='train', transform=None, pre_transform=None, pre_filter=None, ds=None,cold=None):
        self.ds = ds
        self.cold = cold
        super().__init__(root, transform, pre_transform, pre_filter)
        if types == 'train':
            self.data, self.slices = torch.load(self.processed_paths[index])
        elif types == 'test':
            self.data, self.slices = torch.load(self.processed_paths[index + len(self.raw_paths)])

    @property
    # def raw_file_names(self):
    #     return ['kiba_fold_1.csv', 'kiba_fold_2.csv', 'kiba_fold_3.csv',
    #             'kiba_fold_4.csv', 'kiba_fold_5.csv']
    # def raw_file_names(self):
    #     return ['kiba_cold_protein_fold_1.csv', 'kiba_cold_protein_fold_2.csv', 'kiba_cold_protein_fold_3.csv',
    #             'kiba_cold_protein_fold_4.csv', 'kiba_cold_protein_fold_5.csv']

    # def raw_file_names(self):

    #     return ['davis_cold_drug_fold_1.csv', 'davis_cold_drug_fold_2.csv', 'davis_cold_drug_fold_3.csv', 'davis_cold_drug_fold_4.csv', 'davis_cold_drug_fold_5.csv']

    # def raw_file_names(self):

    #     return ['davis_cold_protein_fold_1.csv', 'davis_cold_protein_fold_2.csv', 'davis_cold_protein_fold_3.csv', 'davis_cold_protein_fold_4.csv', 'davis_cold_protein_fold_5.csv']

    # def raw_file_names(self):
    #     return ['davis_cold_pairs_fold_1.csv', 'davis_cold_pairs_fold_2.csv', 'davis_cold_pairs_fold_3.csv',
    #             'davis_cold_pairs_fold_4.csv', 'davis_cold_pairs_fold_5.csv']
    # def raw_file_names(self):
    #     return ['davis_fold_1.csv', 'davis_fold_2.csv', 'davis_fold_3.csv',
    #             'davis_fold_4.csv', 'davis_fold_5.csv']
    # def raw_file_names(self):
    #     return ['kiba_fold_1.csv', 'kiba_fold_2.csv', 'kiba_fold_3.csv',
    #             'kiba_fold_4.csv', 'kiba_fold_5.csv']
    def raw_file_names(self):
        if self.cold != 'feicold':
            return [self.ds + '_cold_' + self.cold + '_fold_1.csv', self.ds + '_cold_' + self.cold + '_fold_2.csv', self.ds + '_cold_' + self.cold + '_fold_3.csv',
                    self.ds + '_cold_' + self.cold + '_fold_4.csv', self.ds + '_cold_' + self.cold + '_fold_5.csv']
        return [self.ds + '_fold_1.csv', self.ds + '_fold_2.csv', self.ds + '_fold_3.csv',
                self.ds + '_fold_4.csv', self.ds + '_fold_5.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_fold_0.pt', 'processed_data_fold_1.pt', 'processed_data_fold_2.pt',
                'processed_data_fold_3.pt', 'processed_data_fold_4.pt',
                'processed_test_data_fold_0.pt', 'processed_test_data_fold_1.pt', 'processed_test_data_fold_2.pt',
                'processed_test_data_fold_3.pt', 'processed_test_data_fold_4.pt']

    def download(self):
        pass


def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            label = data.y
            running_loss.update(loss.item(), label.size(0))
    epoch_loss = running_loss.get_average()
    running_loss.reset()
    model.train()
    return epoch_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='davis/feicold', help='task name')
    # parser.add_argument('--dataset', action='store_true', help='task name')
    # parser.add_argument('--dataset', default='davis/drug', help='task name')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    args = parser.parse_args()

    params = dict(
        data_root="data",
        save_dir="save",
        dataset=args.dataset,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )

    logger = TrainLogger(params)
    logger.info(__file__)
    DATASET = params.get("dataset")
    save_model = params.get("save_model")
    data_root = params.get("data_root")
    batch_size = params.get("batch_size")
    fpath = os.path.join(data_root, DATASET).replace('/', '\\')
    ds, cold = DATASET.split('/')
    if cold == 'pair':
        cold = cold+'s'
    test_set_list = []
    set_list = []

    for index in range(5):
        test_set_list.append(GNNDataset(fpath, types='test', index=index, ds=ds, cold=cold))

        set_list.append(GNNDataset(fpath, types='train', index=index, ds=ds, cold=cold))

    num_workers = 8 if sys.platform == 'linux' else 0

    print('Number of workers: ', num_workers)

    device = torch.device('cuda:0')
    # drug_affinity_graph = torch.load("/home/yanxy/L/secondMGR/data/davis/affinity/drug_affinity_graph.pt",
    #                                  map_location=device)
    # protein_affinity_graph = torch.load("/home/yanxy/L/secondMGR/data/davis/affinity/protein_affinity_graph.pt",
    #                                    map_location=device)
    # drug_affinity_graph = torch.load("/home/yanxy/L/secondMGR/data/kiba/affinity/kiba_drug_affinity_graph.pt",map_location=device)
    # protein_affinity_graph = torch.load("/home/yanxy/L/secondMGR/data/kiba/affinity/kiba_protein_affinity_graph.pt",map_location=device)
    # drug_affinity_graph = torch.load("D:\PythonProject\secondMGR\data\kiba\\affinity\kiba_drug_affinity_graph.pt",
    #                                  map_location=device)
    # protein_affinity_graph = torch.load("D:\PythonProject\secondMGR\data\kiba\\affinity\kiba_protein_affinity_graph.pt",
    #                                     map_location=device)
    affinity = read_data(ds)
    affinity_graph = create_affinity_graph(affinity, ds)

    for i in range(5):
        test_loader = DataLoader(test_set_list[i], batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 drop_last=True)
        concat_dataset = ConcatDataset([it for j, it in enumerate(set_list) if i != j])
        train_loader = DataLoader(dataset=concat_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True,
                                  drop_last=True
                                  )  # 将一个分片作为测试集，其余分片作为训练集，实现交叉验证。

        logger.info('fold_' + str(i))
        logger.info(f"batch size: {batch_size}")
        logger.info(f"Use dataset: {DATASET}")
        logger.info(f"Number of train: {len(train_loader)}")
        logger.info(f"Number of test: {len(test_loader)}")
        logger.info(f"save_model: {save_model}")
        epoch_num = (math.ceil((3000 * 50) / len(train_loader))) * len(train_loader) / 50
        logger.info(f"epochs: {epoch_num}")
        device = torch.device('cuda:0')
        # model = DrugMGR1(drug_affinity_graph=drug_affinity_graph, protein_affinity_graph=protein_affinity_graph).to(
        #     device)
        model = DrugMGR1(affinity_graph=affinity_graph, dataset=ds).to(device)
        # logger.info(model)
        epochs = 3000
        steps_per_epoch = 50
        num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))  # 51
        break_flag = False
        optimizer = optim.Adam(model.parameters(), lr=params.get("lr"))
        criterion = nn.MSELoss()
        global_step = 0
        global_epoch = 0
        early_stop_epoch = 300
        running_loss = AverageMeter()
        running_cindex = AverageMeter()
        running_best_mse = BestMeter("min")
        model.train()
        best_result_msg = ''
        best_model = model
        best_model_name = ''
        for _ in range(num_iter):
            if break_flag:
                break
            for data in train_loader:
                global_step += 1
                # print("global_step: ", global_step)
                device = torch.device('cuda:0')
                data = data.to(device)
                model = model.to(device)
                pred = model(data)
                loss = criterion(pred.view(-1), data.y.view(-1))
                cindex = get_cindex(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss.update(loss.item(), data.y.size(0))
                running_cindex.update(cindex, data.y.size(0))
                if global_step % steps_per_epoch == 0:
                    global_epoch += 1
                    epoch_loss = running_loss.get_average()
                    epoch_cindex = running_cindex.get_average()
                    running_loss.reset()
                    running_cindex.reset()
                    test_loss, test_cindex, test_r2 = test_result(model, criterion, test_loader, device)
                    msg = "epoch-%d, loss-%.4f, cindex-%.4f, test_loss-%.4f, test_cindex-%.4f, test_r2-%.4f" % (
                        global_epoch, epoch_loss, epoch_cindex, test_loss, test_cindex, test_r2)
                    # epoch_msg = "epoch%dtest_loss%.4f" % (global_epoch, test_loss)
                    logger.info(msg)
                    if test_loss < running_best_mse.get_best():
                        best_model_name = "epoch%dtest_loss%.4f" % (global_epoch, test_loss)
                        running_best_mse.update(test_loss)

                        best_result_msg = "test_loss:%.4f, test_cindex:%.4f, test_r2:%.4f" % (
                            test_loss, test_cindex, test_r2)

                        logger.info('######' + best_result_msg + '######')
                        best_model = model
                        # 保存模型,当save_model为True且周期大于倒数300(早停)+100(扩充)epoch时开始进行保存模型。
                        if save_model and global_epoch > epoch_num - early_stop_epoch - 50:
                            save_model_dict(model, logger.get_model_dir(), best_model_name)
                    else:
                        # 连续有early_stop_epoch这么多次没有找到更好的结果test_loss<running_best_mse.get_best()时触发早停，防止过拟合。
                        count = running_best_mse.counter()
                        if count > early_stop_epoch:
                            logger.info(f"early stop in epoch {global_epoch}")
                            break_flag = True
                            logger.info('fold_' + str(i) + "'s result is: " + best_result_msg)
                            if save_model:
                                save_model_dict(best_model, logger.get_model_dir(), best_model_name)
                            break


def test_result(model, criterion, test_loader, device):
    criterion = nn.MSELoss()
    test_loss, test_cindex, test_r2 = test_val(model, criterion, test_loader, device)
    return test_loss, test_cindex, test_r2


def test_val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()
    pred_list = []
    label_list = []
    a = 0
    for data in dataloader:
        # a += 1
        # if a > 50:
        #     break
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            label = data.y
            pred_list.append(pred.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    epoch_cindex = get_cindex(label, pred)
    epoch_r2 = get_rm2(label, pred)
    epoch_loss = running_loss.get_average()
    running_loss.reset()
    return epoch_loss, epoch_cindex, epoch_r2


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


if __name__ == "__main__":
    if sys.platform == 'linux':
        with open(__file__, "r", encoding="utf-8") as f:
            for line in f.readlines():
                print(line)
    main()
