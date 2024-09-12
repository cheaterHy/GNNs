import random

import networkx as nx
import numpy as np
import torch
import pandas as pd
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
from torch import nn, optim
from GCNEmbedding import *
from GATembedding import *
from utils import *
from torch.nn.functional import cosine_similarity
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score,accuracy_score,jaccard_score,normalized_mutual_info_score
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist


color_red = "\033[1;31m"  # 红色字体
color_green = "\033[1;32m"  # 绿色字体
color_reset = "\033[0m"  # 恢复默认颜色


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)  # 设置CPU生成随机数的种子，方便下次复现实验结果。
torch.cuda.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("cuda ", torch.cuda.is_available())

num_fea = -1
data = load_data(args.dataset, num_fea)
# data.x = data.x[:,-1].reshape(-1,1)
data.to(device)

def supervised_select_samples_based_on_edges():

    indices_dic = {}
    for label in range(cluster_number):
        this_label_indices = np.where(data.label.cpu().numpy() == label)[0]
        indices_dic[label] = torch.tensor(this_label_indices)

    distance_matrix = torch.tensor(data.regular_simi)


    topK_indices = {}
    bottomk_indices = {}


    inf_dic = {}
    ninf_dic = {}
    for this_label in list(indices_dic.keys()):
        other_labels = list(indices_dic.keys())
        if this_label in other_labels:
            other_labels.remove(this_label)
        for label in other_labels:
            inf_list = []
            inf_list.append(indices_dic[label])
            inf_dic[label] = set_sample_matrix(distance_matrix,inf_list,float('inf'))
            ninf_dic[label] = set_sample_matrix(distance_matrix, inf_list, float('-inf'))

    for i in range(distance_matrix.shape[0]):
        for label in list(indices_dic.keys()):
            if torch.eq(indices_dic[label],i).any().item():
                _,this_topk = inf_dic[label][i].topk(k=2, largest=True)
                if i not in topK_indices:
                    topK_indices[i] = []
                topK_indices[i].append(this_topk.tolist())
                other_labels = list(indices_dic.keys())
                for other_label in other_labels:
                    _,this_bottomK = ninf_dic[other_label][i].topk(k=2, largest=False)
                    # selected_this_bottomK = random.sample(this_bottomK.tolist(), 2)
                    if i not in bottomk_indices:
                        bottomk_indices[i] = []
                    bottomk_indices[i].extend(this_bottomK.tolist())
                break


    return topK_indices, bottomk_indices

def fix_missing_number(lst):
    n = len(lst)
    for i in range(1, n):
        # 检查当前元素和前一个元素的差值是否为1
        if lst[i] - lst[i-1] != 1:
            # 找到缺失的地方，所有后续元素减1
            for j in range(i, n):
                lst[j] -= 1
            break  # 只需要处理第一个缺失的地方，处理完毕后退出循环
    return lst

def select_samples_based_on_edges():
    degree = data.degree[:,1].view(1,-1)
    degree_sorted_indices = torch.argsort(degree,descending=True)
    top_indices = degree_sorted_indices[:int(degree_sorted_indices.shape[0]*0.01)]
    sec_indices = degree_sorted_indices[int(degree_sorted_indices.shape[0]*0.01):int(degree_sorted_indices.shape[0]*0.05)]
    thr_indices = degree_sorted_indices[int(degree_sorted_indices.shape[0]*0.05):int(degree_sorted_indices.shape[0]*0.2)]
    remain_indiecs = degree_sorted_indices[int(degree_sorted_indices.shape[0]*0.2):]

    diff = degree - degree.T
    distance_matrix = torch.sqrt(diff ** 2)

    pos_numbers = 2
    neg_numbers = 3

    topK_indices = torch.zeros((degree.shape[-1], pos_numbers))
    bottomk_indices = torch.zeros((degree.shape[-1], neg_numbers))

    top_inf = set_sample_matrix(distance_matrix,sec_indices,thr_indices,remain_indiecs,float('inf'))
    sec_inf = set_sample_matrix(distance_matrix,top_indices,thr_indices,remain_indiecs,float('inf'))
    thr_inf = set_sample_matrix(distance_matrix,top_indices,sec_indices,remain_indiecs,float('inf'))
    re_inf = set_sample_matrix(distance_matrix,top_indices,sec_indices,thr_indices,float('inf'))
    top_minf = set_sample_matrix(distance_matrix,sec_indices,thr_indices,remain_indiecs,float('-inf'))
    sec_minf = set_sample_matrix(distance_matrix,top_indices,thr_indices,remain_indiecs,float('-inf'))
    thr_minf = set_sample_matrix(distance_matrix,top_indices,sec_indices,remain_indiecs,float('-inf'))
    re_minf = set_sample_matrix(distance_matrix,top_indices,sec_indices,thr_indices,float('-inf'))

    for i in range(distance_matrix.shape[0]):
        if torch.eq(top_indices,i).any().item():
            _,topK_indices[i] = top_inf[i].topk(k = 2,largest = False)
            _,bottomk_indices[i,0] = sec_minf[i].topk(k = 1, largest = True)
            _,bottomk_indices[i,1] = thr_minf[i].topk(k = 1, largest = True)
            _,bottomk_indices[i,2] = re_minf[i].topk(k = 1, largest = True)
        elif torch.eq(sec_indices,i).any().item():
            _,topK_indices[i] = sec_inf[i].topk(k = 2,largest = False)
            _,bottomk_indices[i,0] = top_minf[i].topk(k = 1, largest = True)
            _,bottomk_indices[i,1] = thr_minf[i].topk(k = 1, largest = True)
            _,bottomk_indices[i,2] = re_minf[i].topk(k = 1, largest = True)
        elif torch.eq(thr_indices,i).any().item():
            _,topK_indices[i] = thr_inf[i].topk(k = 2,largest = False)
            _,bottomk_indices[i,0] = top_minf[i].topk(k = 1, largest = True)
            _,bottomk_indices[i,1] = sec_inf[i].topk(k = 1, largest = True)
            _,bottomk_indices[i,2] = re_minf[i].topk(k = 1, largest = True)
        elif torch.eq(remain_indiecs,i).any().item():
            _,topK_indices[i] = re_inf[i].topk(k = 2,largest = False)
            _,bottomk_indices[i,0] = top_minf[i].topk(k = 1, largest = True)
            _,bottomk_indices[i,1] = sec_minf[i].topk(k = 1, largest = True)
            _,bottomk_indices[i,2] = thr_minf[i].topk(k = 1, largest = True)
        else:
            print('Unexpected false!')
            break

    return topK_indices.long(),bottomk_indices.long()

def set_sample_matrix(distance,other_indices,type):
    diff = distance.clone()
    diff.fill_diagonal_(type)
    for other in other_indices:
        diff[:,other] = type

    return diff

def select_samples_based_on_Reg():

    regular_simi_matrix = data.regular_simi.clone().detach()

    pos_numbers = 3
    neg_numbers = int(data.label.shape[0]*0.02)

    topK_dic = {}
    bottomk_dic = {}

    for i in range(regular_simi_matrix.shape[0]):
        if i not in topK_dic :
            topK_dic[i] = []
        if i not in bottomk_dic :
            bottomk_dic[i] = []
        _,topK_indices = regular_simi_matrix[i].topk(k=pos_numbers,largest=True)
        topK_indices = topK_indices.tolist()
        topK_indices.remove(i)
        _,bottomk_indices = regular_simi_matrix[i].topk(k=neg_numbers,largest=False)
        topK_dic[i].extend(topK_indices)
        bottomk_dic[i].extend(random.sample(bottomk_indices.tolist(),1))

    return topK_dic,bottomk_dic

def compute_clustering_absolute_diff(feautres):
    f = feautres
    abs_sum_vector = torch.sum(torch.abs(f), dim=1, keepdim=True).T
    # abs_sum_vector = torch.max(torch.abs(f),dim=1)

    pos_sets_number = 2
    neg_sets_number = 2

    """Contrastive sets control"""
    if_indices = torch.empty(1)
    nif_indices = torch.empty(1)
    # num_top = int(abs_sum_vector.shape[1] * 0.05)
    _,t_indices = torch.sort(abs_sum_vector,descending=True)
    # if_indices = t_indices[0,:num_top]
    # nif_indices = t_indices[0,num_top:]
    if_mask = torch.zeros(abs_sum_vector.shape[1])
    # if_mask[if_indices] = 1

    diff = abs_sum_vector - abs_sum_vector.T
    distance_matrix = torch.sqrt(diff ** 2)
    distance_matrix.fill_diagonal_(float('-inf')) #除开自身

    up_diff_matrix = distance_matrix.clone()
    down_diff_matrix = distance_matrix.clone()
    # up_diff_matrix[:,nif_indices] = float('-inf')
    # down_diff_matrix[:,if_indices] = float('-inf')


    topK_indices = torch.zeros((distance_matrix.shape[0],pos_sets_number))
    bottomK_indices = torch.zeros((distance_matrix.shape[0],neg_sets_number))

    for i in range(distance_matrix.shape[0]):
        if torch.eq(if_indices, i).any().item():
            _, bottomK_indices[i] = down_diff_matrix[i].topk(k=neg_sets_number,largest=True)
            _, topK_indices[i] = up_diff_matrix[i].topk(k=pos_sets_number,largest=False)
        elif torch.eq(nif_indices, i).any().item():
            _, bottomK_indices[i] = up_diff_matrix[i].topk(k=neg_sets_number,largest=True)
            _,  topK_indices[i]= down_diff_matrix[i].topk(k=pos_sets_number,largest=False)
        else:
            _, bottomK_indices[i] = distance_matrix[i].topk(k=neg_sets_number,largest=True)
            _,  topK_indices[i]= distance_matrix[i].topk(k=pos_sets_number,largest=False)
    topK_indices = topK_indices.long()
    bottomK_indices = bottomK_indices.long()
    return topK_indices,bottomK_indices

def clustering_accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    accuracy = sum(w[row_ind[i], col_ind[i]] for i in range(len(row_ind))) / y_pred.size
    return accuracy

def infoNCE_loss(graph_emb,topk,bottomk):
    loss = 0
    train_indices = np.where(data.train_mask == 1)[0].tolist()
    # train_indices = np.where(tr_id == 1)[0].tolist()
    for i in train_indices:
        positive_sets = graph_emb[topk[i]]
        negative_sets = graph_emb[bottomk[i]]
        pos_cos_similarities = torch.exp(cosine_similarity(graph_emb[i].expand_as(positive_sets), positive_sets, dim=1) / 0.1).sum()
        neg_cos_similarities = torch.exp(cosine_similarity(graph_emb[i].expand_as(negative_sets), negative_sets, dim=1) / 0.1).sum()
        # pos_cos_similarities  = torch.exp(
        #     torch.sqrt(torch.sum((graph_emb[i].expand_as(positive_sets) - positive_sets) ** 2 ,dim=1)) / 0.5).sum()
        # neg_cos_similarities = torch.exp(
        #     torch.sqrt(torch.sum((graph_emb[i].expand_as(negative_sets) - negative_sets) ** 2, dim=1)) / 0.5).sum()
        loss += - torch.log(pos_cos_similarities / (pos_cos_similarities + neg_cos_similarities) )
    loss = loss / len(train_indices)
    # loss += torch.norm(graph_emb[train_indices] - centers[train_indices], p=2)
    loss = loss
    return loss

def KL_loss(graph_emb,centers):
    # q_ij_matrix = torch.zeros((graph_emb.shape[0],centers.shape[0]),dtype=torch.float)
    # for i in range(graph_emb.shape[0]):
    #     for j in range(centers.shape[0]):
    #         q_down = 0
    #         for j_ in range(centers.shape[0]):
    #             q_down +=  torch.norm(graph_emb[i] - centers[j_],p=2)
    #         q_up = 1 / (torch.norm(graph_emb[i] - centers[j],p=2) + 1)
    #         q_ij =  q_up /  (1 /q_down)
    #         q_ij_matrix[i,j]=q_ij
    # f_j = torch.sum(q_ij_matrix,dim=0,keepdim=True)
    # p_down= torch.sum((q_ij_matrix ** 2 / f_j), dim=1,keepdim=True)
    # p_up = q_ij_matrix ** 2 / f_j
    # p_ij_matrix = p_up / p_down.expand_as(p_up)
    # kl_loss =torch.sum( p_ij_matrix * torch.log(p_ij_matrix / q_ij_matrix) )
    # return kl_loss
    distances = torch.zeros((graph_emb.shape[0], centers.shape[0]))
    for i in range(graph_emb.shape[0]):
        for j in range(centers.shape[0]):
            distances[i,j] = torch.norm(graph_emb[i]-centers[j],p=2)
    distances = 1 / (1 + distances)
    distances_sum = torch.sum(distances,dim=1,keepdim=True)
    q_ij_matrix = distances / distances_sum
    f_j = distances.sum(dim = 0)
    p_ij_matrix = (q_ij_matrix ** 2 / f_j) / torch.sum((q_ij_matrix ** 2 / f_j ), dim=1,keepdim=True)
    kl_loss = torch.sum(p_ij_matrix * torch.log(p_ij_matrix / q_ij_matrix))
    return kl_loss,p_ij_matrix

def reconstruction_loss(A,embedding):
    A_hat = F.sigmoid(torch.matmul(embedding, embedding.T))
    r_loss = torch.norm(A - A_hat)
    return r_loss

def train_model(data,topk,bottomk):
    model.train()
    count_loss = float('inf')
    optimizer.zero_grad()
    out = model(data)
    loss = infoNCE_loss(out,topk,bottomk)
    loss.backward()
    optimizer.step()
    print(loss)
    if count_loss > loss:
        count_loss == loss
        torch.save(model.state_dict(), 'model_state_dict.pth')
    return out

def train_clustering_model(data,topk,bottomk,centers):
    model.train()
    count_loss = float('inf')
    out = model(data)
    # centers = model.centers
    # k_input = out.detach().cpu().numpy()

    loss = infoNCE_loss(out,topk,bottomk) + KL_loss(out,centers)[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # if count_loss > loss:
    #     count_loss == loss
    #     torch.save(model.state_dict(), 'clustering_model_state_dict.pth')

    return loss,out

def pre_trian_model(data,topk,bottomk):
    model.train()
    out = model(data)

    edge_index = data.edge_index
    num_nodes = data.num_nodes

    adj_matrix_sparse = torch.sparse_coo_tensor(
        indices=edge_index,
        values=torch.ones(edge_index.shape[1], device=device),
        size=(num_nodes, num_nodes)
    )

    adj_matrix_dense = adj_matrix_sparse.to_dense()
    pre_loss = infoNCE_loss(out,topk,bottomk) + reconstruction_loss(adj_matrix_dense,out)
    optimizer.zero_grad()
    pre_loss.backward()
    optimizer.step()

    return pre_loss


@torch.no_grad()
def test_model(data):
    model.eval()
    # model.load_state_dict(torch.load('clustering_model_state_dict.pth'))
    embeddings = model(data)
    # loss = infoNCE_loss(embeddings, topk, bottomk,centers)
    return embeddings

def min_max_normalize(tensor, min_value=0.0, max_value=1.0):
    """
    对输入张量进行Max-Min归一化，将其缩放到指定范围内。

    参数:
    tensor (torch.Tensor): 需要归一化的张量
    min_value (float): 归一化后的最小值
    max_value (float): 归一化后的最大值

    返回:
    torch.Tensor: 归一化后的张量
    """
    tensor_min = tensor.min()
    tensor_max = tensor.max()

    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    normalized_tensor = normalized_tensor * (max_value - min_value) + min_value

    return normalized_tensor

def plot_clusters(X, labels, true_labels, centroids,test_mask):
    colors = ['blue', 'green']
    markers = ['o', '^']

    plt.figure(figsize=(8, 6))

    for label in np.unique(labels):
        for true_label in np.unique(true_labels):
            mask = (labels == label) & (true_labels == true_label)
            cluster_points = X[test_mask][mask]
            plt.scatter(
                cluster_points[:, 0], cluster_points[:, 1],
                c=colors[label], marker=markers[true_label],
                label=f'Cluster {label}, True {true_label}'
            )

    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x', label='Centroids')

    plt.title(f'{args.dataset} K-means Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def plot_clusters_4(X, labels, true_labels, centroids, test_mask, dataset_name):
    # 增加更多的颜色和标记以支持4个类别
    colors = ['blue', 'green', 'orange', 'purple']
    markers = ['o', '^', 's', 'D']

    plt.figure(figsize=(8, 6))

    # 绘制数据点，根据标签和真实标签组合进行绘制
    for label in np.unique(labels):
        for true_label in np.unique(true_labels):
            mask = (labels == label) & (true_labels == true_label)
            cluster_points = X[test_mask][mask]
            plt.scatter(
                cluster_points[:, -1], cluster_points[:, 1],
                c=colors[label], marker=markers[true_label],
                label=f'Cluster {label}, True {true_label}'
            )

    # 绘制中心点
    plt.scatter(centroids[:, -1], centroids[:, 1], s=300, c='red', marker='x', label='Centroids')

    # 设置标题和标签
    plt.title(f'{dataset_name} K-means Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    """#########  Cluster control  ############"""
    cluster_number = 2
    centriods = torch.rand((cluster_number,128),dtype=torch.float).to(device)


    """#########  training control ###########"""
    pre_train_flag = 1
    train_flag = 1

    """###############   Initial model ##########"""
    if args.type == "Binary":
        # model = GATemb(in_channels=1,out_channels=2,hidden_channels=128).to(device)
        model = GCNemb(num_node_features=data.x.shape[0],output_dim=2,hidden_dim=128,cluster_num=4).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        # optimizer = optim.SGD(model.parameters(),lr=args.learning_rate, weight_decay=args.weight_decay)
    print(model)

    """##############  Sample construction  ###############"""
    # topk, bottomk = supervised_select_samples_based_on_edges()
    topk, bottomk = select_samples_based_on_Reg()


    '''################  Feature reconstruction #############'''
    # data.x = data.added_fea[:,-1].view(-1,1).to(torch.float)
    test_fea = pd.read_csv("../data/{}/{}-cen.csv".format(args.dataset,args.dataset),index_col=False).iloc[:,2:]
    for fea_indices in range(test_fea.shape[1]):
        fea_name = test_fea.columns[fea_indices]
        data.x = torch.tensor(test_fea.iloc[:,fea_indices],dtype=torch.float).view(-1,1).to(device)
        print(fea_name)

        count_loss = float('inf')

        if pre_train_flag == 1:
            for i in range(100):
                pre_loss = pre_trian_model(data,topk,bottomk)
                print(f'pretrain loss: {pre_loss}')
                if count_loss > pre_loss:
                    torch.save(model.state_dict(),'pre_train_model.pth')
            model.load_state_dict(torch.load('pre_train_model.pth'))

        if train_flag == 1:
            for i in range(100):
                loss,out = train_clustering_model(data,topk,bottomk,centers=centriods)
                k_input = out.detach().cpu().numpy()
                kmeans = KMeans(n_clusters=cluster_number)
                kmeans.fit(k_input)
                centriods = torch.tensor(kmeans.cluster_centers_).to(device)
                cluster_results = kmeans.fit_predict(k_input)

                train_nmi = normalized_mutual_info_score(data.label.cpu().numpy(),cluster_results)
                train_acc = clustering_accuracy(data.label.cpu().numpy(),cluster_results)
                print(f"epoch: {color_green}{i}{color_reset}, total_loss: {color_red}{loss}{color_reset}, ACC: {color_red}{train_acc}{color_reset}, NMI: {color_red}{train_nmi}{color_reset}")
            emd = test_model(data)
            emd = emd.cpu().numpy()
            """############# Downstream task ##############"""
            for mode in ['kmeans']:
                if mode =='kmeans':
                    test_data = emd[data.test_mask]
                    test_labels = data.label[data.test_mask]

                    kmeans = KMeans(n_clusters=cluster_number)
                    predict_labels = kmeans.fit_predict(test_data)
                    cen =  kmeans.cluster_centers_

                    merge_flag = 0
                    if merge_flag == 1:
                        for i in range(test_labels.shape[0]):
                            if test_labels[i] == 2:
                                test_labels[i] = 1
                            if test_labels[i] == 3:
                                test_labels[i] = 0

                            if predict_labels[i] == 2:
                                predict_labels[i] = 0
                            if predict_labels[i] == 3:
                                predict_labels[i] = 1

                    influ_acc = False
                    if influ_acc == True:
                        influ_indices = np.where((test_labels.cpu().numpy() == 1 )| (test_labels.cpu().numpy() == 2))
                        influ_predict = predict_labels[influ_indices]
                        values, counts = np.unique(influ_predict, return_counts=True)

                        max_count_index = np.argmax(counts)
                        most_frequent_element = values[max_count_index]
                        max_count = counts[max_count_index]

                        print(f'influential nodes clustering ACC: {float(max_count / influ_predict.shape[0])}')


                    print(args.dataset)
                    print(cen.shape)

                    nmi = normalized_mutual_info_score(test_labels.cpu().numpy(),predict_labels)
                    acc = clustering_accuracy(test_labels.cpu().numpy(),predict_labels)
                    ari = adjusted_rand_score(test_labels.cpu().numpy(),predict_labels)
                    print(f"ACC: {acc}")
                    print(f"NMI: {nmi}")
                    print(f'ARI: {ari}')

                    plot_clusters_4(emd,predict_labels,test_labels.cpu().numpy(),cen,data.test_mask,dataset_name=args.dataset)

                    columns = ['metrics','ACC','NMI',"ARI"]
                    records = pd.DataFrame(columns=columns)
                    records = records.append({'metrics': fea_name, 'ACC': acc, 'NMI': nmi, 'ARI': ari}, ignore_index=True)
                    records.to_csv(f'D:\\temporal save\\{args.dataset}-diffcen.csv',mode='a',index=False)
                if mode =='knn':
                    k = 1
                    knn = KNeighborsClassifier(n_neighbors=k)
                    x_train = emd[data.train_mask]
                    y_train = data.label[data.train_mask].cpu().numpy()
                    knn.fit(x_train,y_train)

                    x_test = emd[data.test_mask]
                    y_test = data.label[data.test_mask].cpu().numpy()
                    y_pred = knn.predict(x_test)
                    acc = accuracy_score(y_test,y_pred)
                    print(mode)
                    print(args.dataset)
                    print(f'Accuracy: {acc:.4f}')
                elif mode == 'svm':
                    clf = SVC(kernel='rbf')
                    x_train = emd[data.train_mask]
                    y_train = data.label[data.train_mask].cpu().numpy()
                    clf.fit(x_train,y_train)

                    x_test = emd[data.test_mask]
                    y_test = data.label[data.test_mask].cpu().numpy()
                    y_pred = clf.predict(x_test)
                    acc = accuracy_score(y_test,y_pred)
                    print(mode)
                    print(args.dataset)
                    print(f'Accuracy: {acc:.4f}')
                elif mode == 'lr':
                    clf = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)
                    x_train = emd[data.train_mask]
                    y_train = data.label[data.train_mask].cpu().numpy()
                    clf.fit(x_train, y_train)

                    x_test = emd[data.test_mask]
                    y_test = data.label[data.test_mask].cpu().numpy()
                    y_pred = clf.predict(x_test)
                    acc = accuracy_score(y_test,y_pred)
                    print(mode)
                    print(args.dataset)
                    print(f'Accuracy: {acc:.4f}')
