import sys
import random
from collections import defaultdict

class COPRAGraph:
    def __init__(self):
        self.adj = defaultdict(list)  # adjacency list
        self.nodes = set()

    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)
        self.nodes.add(u)
        self.nodes.add(v)

    def read_edgelist(self, filename):
        with open(filename, 'r') as f:
            # Optional: skip header if needed. If the first line is a header, uncomment next line.
            f.readline()  
            for line in f:
                parts = line.strip().split()
                # print(parts) # ---------------------------------------------------------------------------------------------------------
                if len(parts) < 2:
                    continue
                u, v = int(parts[0]), int(parts[1])
                self.add_edge(u, v)

class COPRA:
    def __init__(self, filename, v, max_iterations):
        self.graph = COPRAGraph()
        self.graph.read_edgelist(filename)
        self.v = v
        self.max_iterations = max_iterations
        # community_memberships[node] = dict {community_label: weight}
        self.community_memberships = dict()
        self.initialize_communities()

    def initialize_communities(self):
        # Initially each node is its own community
        for node in self.graph.nodes:
            self.community_memberships[node] = {node: 1.0}

    def run(self):
        # Perform label propagation allowing multiple memberships
        nodes = list(self.graph.nodes)
        for iteration in range(self.max_iterations):
            random.shuffle(nodes)
            new_memberships = {}

            for node in nodes:
                # Collect memberships from neighbors
                neighbor_memberships = defaultdict(float)
                if len(self.graph.adj[node]) == 0:
                    # Isolated node keeps its membership
                    neighbor_memberships = self.community_memberships[node].copy()
                else:
                    for neighbor in self.graph.adj[node]:
                        # Add all the communities of the neighbor
                        for c_label, c_weight in self.community_memberships[neighbor].items():
                            neighbor_memberships[c_label] += c_weight

                # Normalize
                total = sum(neighbor_memberships.values())
                for c_label in neighbor_memberships:
                    neighbor_memberships[c_label] /= total

                # If more than v communities, keep top v
                if len(neighbor_memberships) > self.v:
                    # Keep top v by weight
                    sorted_coms = sorted(neighbor_memberships.items(), key=lambda x: x[1], reverse=True)
                    neighbor_memberships = dict(sorted_coms[:self.v])

                new_memberships[node] = neighbor_memberships

            self.community_memberships = new_memberships

    def export_communities(self, filename, verbose=False):
        # After final iteration, communities are determined by grouping nodes by each community label
        # One way is to invert community_memberships to community -> list of nodes
        community_to_nodes = defaultdict(list)
        for node, coms in self.community_memberships.items():
            for c in coms.keys():
                community_to_nodes[c].append(node)

        # Write communities to file
        with open(filename, 'w') as f:
            for c, members in community_to_nodes.items():
                members = sorted(members)
                f.write(f"{members}\n")

        if verbose:
            print(f"{len(community_to_nodes)} communities identified and written to {filename}")

def main():
    if len(sys.argv) < 5:
        print("Usage: python copra.py [input_file] [output_file] [v] [max_iterations]")
        print("input_file: edgelist file")
        print("output_file: file to write communities")
        print("v: max number of communities per node (e.g., 3)")
        print("max_iterations: number of iterations (e.g., 20)")
        sys.exit()

    in_file = sys.argv[1]
    out_file = sys.argv[2]
    v = int(sys.argv[3])
    max_iterations = int(sys.argv[4])

    copra = COPRA(in_file, v, max_iterations)
    copra.run()
    copra.export_communities(out_file)

if __name__ == "__main__":
    main()
