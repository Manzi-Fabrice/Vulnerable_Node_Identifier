import pandas as pd

class DirectedGraph:
    def __init__(self):
        self.graph = {}

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []

    def add_edge(self, src, dest, weight):
        self.add_node(src)
        self.add_node(dest)
        self.graph[src].append((dest, weight))

    def display(self):
        for node in self.graph:
            print(f"{node} -> {self.graph[node]}")

def create_network_graph_from_csv(csv_path, min_flow_duration=10):

    graph = DirectedGraph()
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Filtering short-lived connections
    filtered_df = df[df['Flow Duration'] >= min_flow_duration]

    for _, row in filtered_df.iterrows():
        src_port = row['Source Port']
        dest_port = row['Destination Port']

        fwd_packets = int(row['Total Fwd Packets'])
        bwd_packets = int(row['Total Backward Packets'])
        edge_weight = fwd_packets + bwd_packets

        # Create an edge only if both forward and backward packets exist
        if fwd_packets > 0 and bwd_packets > 0:
            graph.add_edge(src_port, dest_port, edge_weight)

    return graph

path = "/Users/manzifabriceniyigaba/Desktop/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
graph = create_network_graph_from_csv(path)
graph.display()
