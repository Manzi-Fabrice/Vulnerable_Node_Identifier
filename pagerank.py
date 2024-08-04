import numpy as np
from collections import deque


def bfs_build_adjacency_matrix(graph):
    N = len(graph)
    adjacency_matrix = np.zeros((N, N))

    for node in range(N):
        visited = [False] * N
        queue = deque([node])

        while queue:
            current = queue.popleft()
            if not visited[current]:
                visited[current] = True
                for neighbor in graph[current]:
                    adjacency_matrix[node][neighbor] += 1
                    if not visited[neighbor]:
                        queue.append(neighbor)

    return adjacency_matrix

def calculate_transition_matrix(adjacency_matrix):
    N = len(adjacency_matrix)
    transition_matrix = np.zeros((N, N))

    for i in range(N):
        row_sum = np.sum(adjacency_matrix[i])
        if row_sum == 0:
            transition_matrix[i] = np.ones(N) / N  # Handling dangling node
        else:
            transition_matrix[i] = adjacency_matrix[i] / row_sum

    return transition_matrix

def compute_pagerank(transition_matrix, alpha=0.85, tolerance=1e-6, max_iter=100):
    N = transition_matrix.shape[0]
    pr = np.ones(N) / N  # Initialize PageRank vector
    teleport_vector = np.ones(N) / N

    for _ in range(max_iter):
        new_pr = alpha * np.dot(transition_matrix, pr) + (1 - alpha) * teleport_vector
        if np.linalg.norm(new_pr - pr, 1) < tolerance:
            break
        pr = new_pr

    return pr
