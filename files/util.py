import numpy as np
import heapq

def Dijkstra(adjacency_matrix, directed=False):
    """
    Compute shortest paths using Dijkstra's algorithm in a memory-efficient way.

    Parameters:
    - adjacency_matrix: numpy array (m, m), graph adjacency with np.inf for non-edges.
    - directed: bool, whether the graph is directed.

    Returns:
    - geodesic_distances: numpy array (m, m) with shortest path distances.
    """
    m = adjacency_matrix.shape[0]
    geodesic_distances = np.full((m, m), np.inf)
    for start_node in range(m):
        # Priority queue (min-heap): (distance, node)
        min_heap = [(0, start_node)]
        visited = np.zeros(m, dtype=bool)
        distances = np.full(m, np.inf)
        distances[start_node] = 0

        while min_heap:
            current_distance, current_node = heapq.heappop(min_heap)

            if visited[current_node]:
                continue
            visited[current_node] = True

            # Get all neighbors (non-infinity)
            neighbors = np.where(adjacency_matrix[current_node] < np.inf)[0]
            new_distances = current_distance + adjacency_matrix[current_node, neighbors]

            # Only update if new distance is smaller
            update_mask = new_distances < distances[neighbors]
            distances[neighbors[update_mask]] = new_distances[update_mask]

            # Push updated distances to the heap
            for neighbor, new_dist in zip(neighbors[update_mask], new_distances[update_mask]):
                heapq.heappush(min_heap, (new_dist, neighbor))

        geodesic_distances[start_node] = distances

    return geodesic_distances
