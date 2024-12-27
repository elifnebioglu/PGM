import torch
import torch.nn.functional as F
import random
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GAE
from sklearn.mixture import BayesianGaussianMixture
from collections import defaultdict



# Define GCN-based GAE encoder
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)



class GNN:
    def __init__(self, filename, epochs, threshold):
        self.epochs = epochs
        self.threshold = threshold
        edge_list = []
        with open(filename, 'r') as f:
            f.readline() # to skip headers
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                u, v = int(parts[0]), int(parts[1])
                edge_list.append([u, v])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        self.num_nodes = edge_index.max().item() + 1
        x = torch.eye(self.num_nodes)
        self.data = Data(x=x, edge_index=edge_index)

        out_channels = 16
        encoder = Encoder(in_channels=self.data.num_features, out_channels=out_channels)
        self.model = GAE(encoder)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
    
    def train(self, verbose=False):
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            self.optimizer.zero_grad()
            z = self.model.encode(self.data.x, self.data.edge_index)
            loss = self.model.recon_loss(z, self.data.edge_index)
            loss.backward()
            self.optimizer.step()

            if epoch % 5 == 0 and verbose:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss.item()}")
    
    def getCommunities(self):
        # Obtain embeddings for nodes
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(self.data.x, self.data.edge_index)
        z_np = z.detach().cpu().numpy()

        # Apply Bayesian GMM
        bgmm = BayesianGaussianMixture(n_components=min(self.num_nodes // 10, 1000))
        labels = bgmm.fit_predict(z_np)

        # Organize communities
        self.community_dict = defaultdict(list)
        for node_id, label in enumerate(labels):
            self.community_dict[label].append(node_id)
        
        return
    
    def run(self):
        self.train()
        self.getCommunities()
    
    def export_communities(self, filename, verbose=False):
        with open(filename, 'w') as f:
            for cid, nodes in self.community_dict.items():
                f.write(f"{nodes}\n")

        if verbose:
            print(f"A total of {len(self.community_dict.keys())} communities were written to: {filename}")



def main():
    infile = '../realGraphs/football_edge_list.txt'
    outfile = 'testGNN.txt'

    gnn = GNN(infile, 5, 0.5)
    gnn.run()
    gnn.export_communities(outfile)

if __name__ == "__main__":
    main()