# init the vertexai package
import vertexai
# Load the text embeddings model
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from tqdm import tqdm
import time

# ACS packages
import networkx as nx
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


LOCATION = "us-central1"
PROJECT_ID = "synthetic-data-432701"

# Embedding Function
def get_embeddings_task(texts, task='CLUSTERING', batch_size=32):
    '''
    Get embeddings for a list of texts with a specific task
    task = ;'CLUSTERING' or 'SEMANTIC_SIMILARITY'
    '''
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        inputs = [TextEmbeddingInput(text, task) for text in texts[i : i + batch_size]]
        batch_embs = model.get_embeddings(inputs)
        embs.extend([embedding.values for embedding in batch_embs])
    return embs

# ACS Execution
def get_acs_k(cos_sim, labels, K, max_degree=None, sim_lb=0, coverage=0.9):
    _, _, selected_samples, _ = binary_similarity_search(cos_sim, K, coverage, labels=labels, max_degree=max_degree, sim_lb=sim_lb)
    return selected_samples



# --- ACS Utilties --- #
def build_graph(cos_sim, sim_thresh=0.0, max_degree=None, labels=None):
    """
    Builds a graph from cosine similarity matrix, keeping only the edges of highest similarity up to max_degree.

    Args:
        cos_sim: A 2D list or numpy array representing the cosine similarity matrix.
        sim_thresh: Minimum similarity threshold for edge creation.
        max_degree: Maximum degree of a node. If provided, keeps only the highest similarity edges.
        labels: A list of labels for each node. If provided, only creates edges between nodes with the same label.

    Returns:
        A networkx Graph object.
    """
    G = nx.Graph()
    for i in range(len(cos_sim)):
        G.add_node(i)
        # Sort neighbors by similarity in descending order
        neighbors = sorted(enumerate(cos_sim[i]), key=lambda x: x[1], reverse=True)
        
        for j, similarity in neighbors:
            if j == i:
                continue
            # if max_degree and added_edges >= max_degree:
            #     break  # Exit the inner loop if max_degree is reached
            if similarity >= sim_thresh and (labels is None or labels[i] == labels[j]):
                G.add_edge(i, j, weight=similarity)

    # Prune edges
    if max_degree is not None:
        for i in range(len(cos_sim)):
            if G.neighbors(i) is None: continue
            neighbors = sorted([n for n in G.neighbors(i) if n != i],  # List neighbors, excluding self
                key=lambda n: G[i][n]['weight'],        # Sort key is the edge weight
                reverse=True                            # Sort in descending order
            )

            diff = len(neighbors) - max_degree
            if diff > 0:
                last_k_nodes = neighbors[diff:]
                edges_to_remove = []
                for j in last_k_nodes:
                    edges_to_remove.append((i, j))
                G.remove_edges_from(edges_to_remove)

    # add self-loop, doesn't count toward max_degree
    for i in range(len(cos_sim)):    
        G.add_edge(i, i, weight=1)
    return G



# Graph sampling algorithms (max-cover)
def max_cover_sampling(graph, k):
    nodes = list(graph.nodes())
    selected_nodes = set()
    covered_nodes = set()

    for _ in range(k):
      if not nodes:
        break
      max_cover_node = max([node for node in nodes if node not in covered_nodes], key=lambda n: len(set(graph.neighbors(n)) - covered_nodes))
      selected_nodes.add(max_cover_node)
      covered_nodes.update(graph.neighbors(max_cover_node))

      # Remove neighbors of selected node
      for neighbor in graph.neighbors(max_cover_node):
        if neighbor in nodes:
          nodes.remove(neighbor)
    return list(selected_nodes), len(nodes)

def binary_similarity_search(data, num_samples, coverage, max_degree=None, epsilon=None, labels=None, sim_lb=707):
    total_num = len(data)

    if epsilon is None:
        # There is a chance that we never get close enough to "coverage" to terminate
        # the loop. I think at the very least we should have epsilon > 1/total_num.
        # So let's set set epsilon equal to the twice of the minimum possible change
        # in coverage.
        epsilon = 5 * (10 / total_num)  # Dynamic epsilon

    if coverage < num_samples / total_num:
        node_graph = build_graph(data, sim_thresh=1.0)   
        samples, rem_nodes = max_cover_sampling(node_graph, num_samples)
        return 1, node_graph, samples, (total_num - rem_nodes) / total_num

    # using an integer for sim threhsold avoids lots of floating drama!
    sim_upper = 1000
    sim_lower = sim_lb
    max_run = 20
    count = 0
    current_coverage = 0
    # Set sim to sim_lower to run the first iteration with sim_lower. If we
    # cannot achieve the coverage with sim_lower, then return the samples.
    sim = (sim_upper + sim_lower) / 2
    if not max_degree:
        max_degree = (2 * total_num * coverage) / num_samples

    while abs(current_coverage - coverage) > epsilon and sim_upper - sim_lower > 1:
        if count >= max_run:
            print(f"Reached max number of iterations ({max_run}). Breaking...")
            break
        count += 1

        node_graph = build_graph(data, sim / 1000, max_degree=max_degree, labels=labels)
        samples, rem_nodes = max_cover_sampling(node_graph, num_samples)
        current_coverage = (total_num - rem_nodes) / total_num

        if current_coverage < coverage:
            sim_upper = sim
        elif coverage == 1.0 and len(samples) < num_samples:
            sim_lower = sim # Handling tricky case
            current_coverage = 0
        else:
            sim_lower = sim
        sim = (sim_upper + sim_lower) / 2

    print(f"Completed with similarity threshold = {sim/1000}")
    return sim / 1000, node_graph, samples, current_coverage