import csv
import torch
from torch_geometric.data import Data
import os
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Dataset,InMemoryDataset
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import argparse
from torch_geometric.nn import GATConv
from torch_geometric.nn import GATv2Conv
import torch_geometric.transforms as T
from torch.nn.functional import normalize
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from numpy import asarray
from sklearn.preprocessing import OneHotEncoder as OneHot

device = torch.device('cuda')
def readAllFileNames(text):
    arr = os.listdir(text)
    return arr


def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)
    return x, mapping

class IdentityEncoder(object):
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return normalize(torch.from_numpy(df.values).view(-1, 1).to(torch.float),p=5.0).to(self.dtype)
class OneHotEncoder(object):
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return F.one_hot(torch.from_numpy(df.values)).to(self.dtype)
def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])
    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)
    return edge_index, edge_attr
def lodder(a,b,text):
    vertices_path = text+"/"+b
    vertices_x, vertices_mapping = load_node_csv(
        vertices_path, index_col='atom_index', encoders={
            'atom_type': OneHotEncoder(dtype=torch.float), ##one hot encoding
            'residue_index': IdentityEncoder(dtype=torch.float),
            'residue_type': OneHotEncoder(dtype=torch.float), ##one hot encoding
            'residue_surface_class': IdentityEncoder(dtype=torch.float),
        })
    vertices_label, vertices = load_node_csv(
        vertices_path, index_col='atom_index', encoders={
            'residue_binding_class': IdentityEncoder(dtype=torch.long)
        })
    edge_path = text+"/"+a
    edge_index, edge_label = load_edge_csv(
        edge_path,
        src_index_col='atom_index1',
        src_mapping=vertices_mapping,
        dst_index_col='atom_index2',
        dst_mapping=vertices_mapping,
        encoders={'interaction_class': IdentityEncoder(dtype=torch.long)},
    )
    a = []
    for x in vertices_label:
        for y in x:
            a.append(y)
    xs = torch.as_tensor(a)
    return Data(x=vertices_x,edge_index=edge_index,edge_attr=edge_label,y=xs)
class MyOwnDataset1(InMemoryDataset):
    def __init__(self,root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load("tensor.pt")
    def savedataset(self):
        arr = readAllFileNames("training")
        data_list = []
        for i in range(int(len(arr)/2)):
            data = lodder(arr[i*2],arr[i*2+1],"training")
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), 'tensor.pt')
class MyOwnDataset2(InMemoryDataset):
    def __init__(self,root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load("tensor2.pt")
    def savedataset(self):
        arr = readAllFileNames("validation")
        data_list = []
        for i in range(int(len(arr)/2)):
            data = lodder(arr[i*2],arr[i*2+1],"validation")
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), 'tensor2.pt')
class MyOwnDataset3(InMemoryDataset):
    def __init__(self,root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load("tensor3.pt")
    def savedataset(self):
        arr = readAllFileNames("testing")
        data_list = []
        for i in range(int(len(arr)/2)):
            data = lodder(arr[i*2],arr[i*2+1],"testing")
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), 'tensor3.pt')
dataset = MyOwnDataset1('.')
dataset.savedataset()
dataset = MyOwnDataset1('.')
validation_dataset = MyOwnDataset2('.')
validation_dataset.savedataset()
validation_dataset = MyOwnDataset2('.')
test_dataset = MyOwnDataset3('.')
test_dataset.savedataset()
test_dataset = MyOwnDataset3('.')
data = dataset[0]
data.to(device)
num_hidden_nodes = 64
num_node_features = data.num_node_features
num_classes = 2

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_node_features, num_hidden_nodes)
        self.conv2 = GCNConv(num_hidden_nodes, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index,data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
class Net2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv(num_node_features, 16, heads=16,
                             dropout=0.1).jittable()
        self.conv3 = GATv2Conv(256, 8, heads=8,
                               dropout=0.1).jittable()
        self.conv2 = GATv2Conv(64, num_classes, heads=1,
                             dropout=0.1).jittable()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index,data.edge_attr
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv3(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

net = Net().to(device)
net2 = Net2().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(net2.parameters(), lr=0.001)

my_parser = argparse.ArgumentParser(description='List the content of a folder')
my_parser.add_argument('number',
                       metavar='number',
                       type=int,
                       help='number')
def train3():
    loader = DataLoader(dataset, batch_size=5, shuffle=True)
    loader2 = DataLoader(test_dataset, batch_size=5, shuffle=True)
    net2.train()
    c = []
    e = []
    lo = []
    for i in range(1000):
        lossList = []
        for batch in loader:
            batch.to(device)
            transform = T.ToUndirected()
            batch = transform(batch)
            output = net2(batch)
            loss = F.cross_entropy(output, batch.y)
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            lossList.append(loss.item())
        loader = DataLoader(validation_dataset, batch_size=5, shuffle=True)
        a = 0
        b = 0
        d = 0
        lol = sum(lossList) / len(lossList)
        lo.append(lol)
        for batch in loader2:
            batch.to(device)
            transform = T.Compose([T.ToUndirected()])
            batch = transform(batch)
            pred = net2(batch).argmax(dim=1)
            d = d + abs((f1_score(batch.y.cpu().numpy(), pred.cpu().numpy())))
            b = b + abs((f1_score(batch.y.cpu().numpy(), pred.cpu().numpy())))
            a = a + abs((matthews_corrcoef(batch.y.cpu().numpy(), pred.cpu().numpy())))
        print(a/len(loader2))
        c.append(b/len(loader2))
        e.append(d / len(loader2))
    print(c)
    print(e)
    print(lo)

def test3():
    loader = DataLoader(test_dataset, batch_size=5, shuffle=True)
    net2.eval()
    a = 0
    d = 0
    for batch in loader:
        batch.to(device)
        transform = T.Compose([T.ToUndirected()])
        batch = transform(batch)
        pred = net2(batch).argmax(dim=1)
        d = d + abs((f1_score(batch.y.cpu().numpy(), pred.cpu().numpy())))
        a = a + abs((matthews_corrcoef(batch.y.cpu().numpy(), pred.cpu().numpy())))
    print(a/len(loader))
    print(d/len(loader))
args = my_parser.parse_args()
input = args.number
if input == 1:
    train()
    test()
if input == 2:
    train3()
    test3()