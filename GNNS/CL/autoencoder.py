import torch.nn
import random
from config import *
from CL.train import KL_loss,clustering_accuracy
from utils import *
import itertools
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import normalized_mutual_info_score ,adjusted_rand_score


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)  # 设置CPU生成随机数的种子，方便下次复现实验结果。
torch.cuda.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("cuda ", torch.cuda.is_available())

numfea = 3
data = load_data(args.dataset,numfea)
data.to(device)

class encoder(torch.nn.Module):
    def __init__(self,input_dim,output_dim):
        super(encoder, self).__init__()
        self.mlp1 = torch.nn.Linear(input_dim,64)
        self.mlp2 = torch.nn.Linear(64,8)
        self.mlp3 = torch.nn.Linear(8,8)
        self.mlp4 = torch.nn.Linear(8,128)
        self.mlp5 = torch.nn.Linear(128,output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self,data):
        x = data
        x = self.relu(self.mlp1(x))
        x = self.relu(self.mlp2(x))
        x = self.relu(self.mlp3(x))
        x = self.relu(self.mlp4(x))
        x = self.mlp5(x)
        return x

class decoder(torch.nn.Module):
    def __init__(self,input_dim,output_dim):
        super(decoder, self).__init__()
        self.mlp1 = torch.nn.Linear(input_dim,128)
        self.mlp2 = torch.nn.Linear(128,8)
        self.mlp3 = torch.nn.Linear(8,8)
        self.mlp4 = torch.nn.Linear(8,64)
        self.mlp5 = torch.nn.Linear(64,output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self,data):
        x = data
        x = self.relu(self.mlp1(x))
        x = self.relu(self.mlp2(x))
        x = self.relu(self.mlp3(x))
        x = self.relu(self.mlp4(x))
        x = self.mlp5(x)
        return x


def compute_directional_centrality(data, k=10):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
    _, indices = nbrs.kneighbors(data)

    directional_centrality = []
    for i in range(len(data)):
        neighbors = data[indices[i][1:]]
        vectors = neighbors - data[i]

        mean_direction = np.mean(vectors, axis=0)
        norm = np.linalg.norm(mean_direction)

        if norm > 0:
            mean_direction /= norm

        centrality = np.sum(mean_direction)
        directional_centrality.append(centrality)

    return np.array(directional_centrality)

def ldcca_clustering(data, k=10, threshold=0.5):

    centrality = compute_directional_centrality(data, k=k)
    labels = -1 * np.ones(len(data),dtype=np.int64)

    cluster_id = 0
    for i, c in enumerate(centrality):
        if c > threshold:
            labels[i] = cluster_id
            cluster_id += 1

    return labels

def l2_loss(x,y):
    return torch.norm((x-y),p=2)
def pre_train():
    encoder_model.train()
    decoder_model.train()
    embeddings = encoder_model(data.x)
    reconstrution = decoder_model(embeddings)
    loss = l2_loss(data.x,reconstrution)

    pre_optim.zero_grad()
    loss.backward()
    pre_optim.step()

    return embeddings,loss

def train():
    encoder_model.load_state_dict(torch.load('saved_model\\auto_encoder.pth'))
    encoder_model.train()
    embeddings = encoder_model(data.x[data.train_mask])
    loss = KL_loss(graph_emb=embeddings,centers=centers)[0]
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss

def test():
    encoder_model.eval()
    embeddings = encoder_model(data.x[data.test_mask])
    p_ij_matrix = KL_loss(graph_emb=embeddings,centers=centers)[1]
    return embeddings,p_ij_matrix


if __name__ == '__main__':

    """
        Initial settings 
    """
    encoder_model = encoder(data.x.shape[0],2).to(device)
    decoder_model = decoder(2,data.x.shape[0]).to(device)
    cluster_number = 2
    kmeans = KMeans(n_clusters=2)
    init_cluster = torch.tensor(kmeans.fit_predict(data.x[:, :cluster_number].cpu().numpy())).to(device)
    centers = torch.tensor(kmeans.cluster_centers_).to(device)
    pre_optim = torch.optim.SGD(itertools.chain(encoder_model.parameters(), decoder_model.parameters()),
                                lr=args.learning_rate, weight_decay=args.weight_decay)
    optim = torch.optim.SGD(itertools.chain(encoder_model.parameters(), centers), lr=args.learning_rate,
                            weight_decay=args.weight_decay)

    """  
        Training control   
    """
    pre_train_flag = 1
    train_flag = 1
    method = 'LDCCA'

    if method == 'DEC':
        if pre_train_flag == 1:
            for i in range(500):
                _,loss = pre_train()
                print(f'pretraining loss: {loss}')
            torch.save(encoder_model.state_dict(),'saved_model\\auto_encoder.pth')

        if train_flag == 1:
            for i in range(100):
                kl_loss = train()
                print(f'kl loss: {kl_loss}')

        embeddings,p_ij_matrix = test()
        predicts = torch.argmax(p_ij_matrix,dim=1).cpu().numpy()
        acc = clustering_accuracy(data.label[data.test_mask],predicts)
        NMI = normalized_mutual_info_score(data.label[data.test_mask].cpu().numpy(),predicts)
        ARI = adjusted_rand_score(data.label[data.test_mask].cpu().numpy(),predicts)
        print(args.dataset)
        print(f'cluster ACC: {acc}, NMI: {NMI}, ARI: {ARI}')
    if method == 'DNC':
        optim = torch.optim.SGD(itertools.chain(encoder_model.parameters(),decoder_model.parameters(),centers), lr=args.learning_rate,
                            weight_decay=args.weight_decay)
        for i in range(1000):
            encoder_model.train()
            decoder_model.train()
            embeddings = encoder_model(data.x[data.train_mask])
            reconstrution = decoder_model(embeddings)
            loss = l2_loss(data.x[data.train_mask], reconstrution)

            distances = torch.sqrt(((embeddings[:, None, :] - centers[None, :, :]) ** 2).sum(dim=2))
            cluster_indices = torch.argmin(distances,dim=1)
            selected_centers = centers[cluster_indices]
            c_loss = torch.norm(embeddings-selected_centers)

            loss = loss + c_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            print(f'training loss: {loss}')
        encoder_model.eval()
        embeddings = encoder_model(data.x[data.test_mask])
        distances = torch.sqrt(((embeddings[:, None, :] - centers[None, :, :]) ** 2).sum(dim=2))
        cluster_indices = torch.argmin(distances, dim=1)
        predicts = cluster_indices.cpu().numpy()
        acc = clustering_accuracy(data.label[data.test_mask],predicts)
        NMI = normalized_mutual_info_score(data.label[data.test_mask].cpu().numpy(),predicts)
        ARI = adjusted_rand_score(data.label[data.test_mask].cpu().numpy(),predicts)
        print(args.dataset)
        print(f'cluster ACC: {acc}, NMI: {NMI}, ARI: {ARI}')
    if method == 'NMF':
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        adj_matrix_sparse = torch.sparse_coo_tensor(
            indices=edge_index,
            values=torch.ones(edge_index.shape[1], device=device),
            size=(num_nodes, num_nodes)
        )
        adj_matrix_dense = adj_matrix_sparse.to_dense()

        X = adj_matrix_dense.cpu().numpy()
        nmf = NMF(n_components=10, init='random', random_state=args.seed)
        W = nmf.fit_transform(X)
        H = nmf.components_
        kmeans = KMeans(n_clusters=5, random_state=0).fit(W)
        predicts = kmeans.labels_
        acc = clustering_accuracy(data.label[data.test_mask],predicts[data.test_mask])
        NMI = normalized_mutual_info_score(data.label[data.test_mask].cpu().numpy(),predicts[data.test_mask])
        ARI = adjusted_rand_score(data.label[data.test_mask].cpu().numpy(),predicts[data.test_mask])
        print(args.dataset)
        print(f'cluster ACC: {acc}, NMI: {NMI}, ARI: {ARI}')
    if method == 'LDCCA':
        predicts = ldcca_clustering(data.x[data.test_mask].cpu().numpy(), k=10, threshold=0.5)
        acc = clustering_accuracy(data.label[data.test_mask].cpu().numpy(),predicts)
        NMI = normalized_mutual_info_score(data.label[data.test_mask].cpu().numpy(),predicts)
        ARI = adjusted_rand_score(data.label[data.test_mask].cpu().numpy(),predicts)
        print(args.dataset)
        print(f'cluster ACC: {acc}, NMI: {NMI}, ARI: {ARI}')