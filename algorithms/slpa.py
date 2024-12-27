
import random
import sys

class Node:
    # Initializes a new node
    def __init__(self, label):
        self.label = label
        self.memory = dict()
        self.memory[label] = 1
        self.memory_count = 1
 
    def send(self):
        sum = 0.0
        r = random.random()
        keys = self.memory.keys()
        for key in keys:
            sum = sum + float(self.memory[key]) / float(self.memory_count)
            if sum >= r:
                return key

class Graph:
    def __init__(self):
        self.node_list = dict()
        self.edge_list = dict()
        self.community_list = dict()

    def add_node(self, label):
        node = Node(label)
        self.node_list[label] = node
        self.edge_list[label] = []

    def get_nodes(self):
        return self.edge_list.keys()

    def add_edge(self, src, dst):
        if src not in self.edge_list:
            self.add_node(src)
        if dst not in self.edge_list:
            self.add_node(dst)
        self.edge_list[src].append(dst)
        self.edge_list[dst].append(src)

    def get_edges(self):
        edges = []
        for src in self.edge_list.keys():
            dst_list = self.edge_list[src]
            for dst in dst_list:
                edge = [src, dst]
                edges.append(edge)
        return edges

    def get_neighbors(self, node):
        return self.edge_list[node]

    # print the edgelist for the graph
    def print_graph(self):
        for node, neighbors in self.edge_list.items():
            print(node, neighbors)

    # print the current state of the memory for each node
    def print_memory(self):
        for node in self.node_list.values():
            print(node.label, node.memory)

    def export_communities(self, filename):
        with open(filename, 'w') as file:
            for key in self.community_list.keys():
                file.write('%s\n' % sorted(self.community_list[key]))
        num_comms = len(self.community_list.keys())
    
    def import_edges(self, filename):
        with open(filename, 'r') as file:
        # Skip the first line if it's a header
            file.readline()

            for line in file:
                words = line.split()  # This will split based on any whitespace (tab, spaces, etc.)
                src = int(words[0].strip())
                dst = int(words[1].strip())
                self.add_edge(src, dst)

        
class SLPA:
    def __init__(self, filename, num_iterations, threshold):
        self.graph = Graph()
        self.graph.import_edges(filename)
        self.num_iterations = num_iterations
        self.threshold = threshold
        
    def run(self, verbose=False):
        nodes = list(self.graph.node_list.keys())
        sys.stdout.write('[')
        for i in range(self.num_iterations):
            if verbose:
                sys.stdout.write('=')
                sys.stdout.flush()
            random.shuffle(nodes)
            self.propagate(nodes)
        if verbose:
            print('] Label Propagation Complete')
        self.post_process()      

    def propagate(self, random_nodes):
        for random_node in random_nodes:
            messages = dict()
            most_popular_message = -1
            messages[most_popular_message] = -1 # added
            listener = self.graph.node_list[random_node]
            neighbors = self.graph.edge_list[random_node]

            for neighbor in neighbors:
                sender = self.graph.node_list[neighbor]
                message = sender.send()

                if message not in messages:
                    messages[message] = 1
                else:
                    messages[message] += 1

                if messages[message] > messages[most_popular_message]: # changed
                    most_popular_message = message

            # Write the most common message received to listener memory
            if most_popular_message != -1:
                if most_popular_message not in listener.memory:
                    listener.memory[most_popular_message] = 1
                else:
                    listener.memory[most_popular_message] += 1
                listener.memory_count += 1
    
    def post_process(self):
        for node in self.graph.node_list.values():
            keys_to_remove = []
            for key in node.memory.keys():
                probability = float(node.memory[key]) / float(node.memory_count)
                if probability < self.threshold:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                node.memory.pop(key, None)

        for node in self.graph.node_list.values():
            for key in node.memory.keys():
                if key not in self.graph.community_list:
                    self.graph.community_list[key] = set()
                self.graph.community_list[key].add(node.label)

    # added
    def export_communities(self, filename):
        self.graph.export_communities(filename)

def verification(graph):
    nodes_v = dict()
    for n in range(1000):
        nodes_v[n]=0

    for node in graph.node_list.keys():
        for n in graph.node_list[node].memory.keys():
            nodes_v[n]=1
    s=0
    for n in nodes_v.keys():
        s+=nodes_v[n]
    return s
    
def main():
    if len(sys.argv) == 2 and sys.argv[1] == '-h':
        print()
        print("USAGE: python slpa.py ['input file'] ['output file'] [num_iterations] [threshold]")
        print('filename: existing file containing edgelist (space-separated endpoints)')
        print('num_iterations: number of iterations for the SLPA label propagation e.g. 20')
        print('threshold: minimum probability density for community membership e.g. 0 - 0.5')
        print()
        sys.exit()

    if not len(sys.argv) == 5:
        print("USAGE: python slpa.py ['input file'] ['output file'] [num_iterations] [threshold]")
        print('for help, type python slpa.py -h')
        sys.exit()

    in_file = sys.argv[1]
    out_file = sys.argv[2]
    num_iterations = int(sys.argv[3])
    threshold = float(sys.argv[4])
 
    slpa = SLPA(in_file, num_iterations, threshold)
    slpa.run(verbose=True)
    slpa.graph.export_communities(out_file)
    print(verification(slpa.graph))



if __name__ == "__main__":
    main()
