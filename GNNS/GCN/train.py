import random
import time
import os

from pytorchtools import EarlyStopping
from matplotlib import pyplot as plt
from sklearn import metrics
from torch import nn, optim
from model import *
from utils import *
from loguru import logger
import pandas as pd
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)  # random seeds
torch.cuda.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("cuda ", torch.cuda.is_available())


def train_binary(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.label[data.train_mask])
    loss.backward()
    optimizer.step() 
    y_pred = out[data.train_mask].argmax(dim=1).cpu().numpy()
    y_test = data.label[data.train_mask].cpu().numpy()
    train_acc = metrics.accuracy_score(y_test, y_pred)
    return train_acc, loss.item()


@torch.no_grad()
def tst_binary(data):
    model.eval()
    accs = []
    pred = model(data)

    loss = F.nll_loss(pred[data.test_mask], data.label[data.test_mask])
    y_pred = pred[data.test_mask].argmax(dim=1).cpu().numpy()
    y_test = data.label[data.test_mask].cpu().numpy()
    F1 = metrics.f1_score(y_pred, y_test)
    accus = metrics.accuracy_score(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    accs.append(accus)
    accs.append(F1)
    accs.append(auc)
    accs.append(fpr)
    accs.append(tpr)

    return accs, loss.item()


if __name__ == '__main__':
    for num_fea in range(1,11,1):
        #load data
        data = load_data(args.dataset, num_fea)
        data.to(device)

        if args.type == "Binary":
            # model = GCN(num_node_features=data.x.shape[1], hidden_dim=args.hidden_units, num_classes=2).to(device)
            model = GCN(num_node_features=num_fea, hidden_dim=16, num_classes=2).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        print(model)

        curtime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        logger.add('Logs\\{}_{}_{}_{}.log'.format(args.dataset, args.feature_type, args.CGCN, curtime))
        time_consumption = 0.0
        losses = []
        losses1 = []
        logger.info(args)
        logger.info(data)
        early_stopping = EarlyStopping(patience=args.early_stopping, verbose=True)

        #save best results
        best_result = {'feature_num' : int(num_fea),
                       'train_loss':float('inf'),
                        'train_acc':float('inf'),
                       'best_loss':float('inf'),
                       'acc':0.0,
                       'F1':0.0,
                       'auc':0.0,
                       'time':0.0}
        for epoch in range(args.epochs):
            if args.type == "Binary":
                time_start = time.time()
                train_accu, loss = train_binary(data)
                accs, test_loss = tst_binary(data)
                time_end = time.time()
                time_consumption = time_consumption + (time_end - time_start)
                losses.append(loss)
                early_stopping(test_loss, model)
                losses1.append(test_loss)
                logger.info(
                    f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_accu:.4f},  Test_Loss: {test_loss:.4f},Test '
                    f'Acc: {accs[0]:.4f},Test F1: {accs[1]:.4f},'
                    f'Test auc: {accs[2]:.4f}')
                if best_result['best_loss'] > test_loss:
                    best_result['train_loss'] = loss
                    best_result['train_acc'] = train_accu
                    best_result['acc'] = accs[0]
                    best_result['F1'] = accs[1]
                    best_result['auc'] = accs[2]
                    best_result['best_loss'] = test_loss
                # early stop
                if early_stopping.early_stop:
                    print("Early stopping")
                    #
                    break
                # save model
                model.load_state_dict(torch.load('checkpoint.pt'))
        logger.info(time_consumption)
        best_result['time'] = time_consumption
        print(best_result)





