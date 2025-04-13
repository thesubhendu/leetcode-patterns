"""
Graph Algorithm Patterns

Graphs are collections of nodes (vertices) connected by edges. They can be:
- Directed or undirected
- Weighted or unweighted
- Cyclic or acyclic

Common graph representations:
1. Adjacency matrix: 2D array where matrix[i][j] indicates edge between i and j
2. Adjacency list: Array of lists where list[i] contains all neighbors of node i

Key graph algorithms:
- Traversal (DFS, BFS)
- Shortest path (Dijkstra's, Bellman-Ford, Floyd-Warshall)
- Minimum spanning tree (Kruskal's, Prim's)
- Topological sort
- Strongly connected components
- Union find (disjoint sets)
"""

from collections import defaultdict, deque
import heapq

# Example 1: Graph Representation and Simple Traversals
class Graph:
    """
    Basic graph implementation using adjacency list.
    """
    def __init__(self, is_directed=False):
        self.adj_list = defaultdict(list)
        self.is_directed = is_directed
    
    def add_edge(self, u, v):
        """Add edge from u to v."""
        self.adj_list[u].append(v)
        # If undirected, add edge from v to u as well
        if not self.is_directed:
            self.adj_list[v].append(u)
    
    def dfs(self, start):
        """
        Depth-First Search traversal starting from 'start' vertex.
        
        Time complexity: O(V + E) where V is number of vertices and E is number of edges
        Space complexity: O(V) for the visited set and recursion stack
        """
        visited = set()
        result = []
        
        def dfs_recursive(vertex):
            if vertex in visited:
                return
            visited.add(vertex)
            result.append(vertex)
            for neighbor in self.adj_list[vertex]:
                dfs_recursive(neighbor)
        
        dfs_recursive(start)
        return result
    
    def bfs(self, start):
        """
        Breadth-First Search traversal starting from 'start' vertex.
        
        Time complexity: O(V + E)
        Space complexity: O(V)
        """
        visited = set([start])
        queue = deque([start])
        result = []
        
        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            
            for neighbor in self.adj_list[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return result

# Example 2: Dijkstra's Algorithm - Shortest Path (LeetCode 743)
def network_delay_time(times: list[list[int]], n: int, k: int) -> int:
    """
    Find the time it takes for all nodes to receive the signal starting from node k.
    
    times[i] = [u, v, w] represents an edge from u to v with weight w (time).
    
    Time complexity: O(E log V) where E is number of edges and V is number of vertices
    Space complexity: O(V + E)
    """
    # Build adjacency list with weights
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    
    # Initialize distances
    distances = {node: float('inf') for node in range(1, n + 1)}
    distances[k] = 0
    
    # Priority queue for Dijkstra's algorithm
    # (distance, node)
    priority_queue = [(0, k)]
    
    # Track visited nodes
    visited = set()
    
    while priority_queue and len(visited) < n:
        current_dist, current_node = heapq.heappop(priority_queue)
        
        # Skip if already visited
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        # Process all neighbors
        for neighbor, weight in graph[current_node]:
            if neighbor not in visited:
                # Calculate new distance
                new_dist = current_dist + weight
                
                # Update if new distance is shorter
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(priority_queue, (new_dist, neighbor))
    
    # Check if all nodes are reachable
    max_distance = max(distances.values())
    return max_distance if max_distance != float('inf') else -1

# Example 3: Union Find - Disjoint Sets (LeetCode 323: Number of Connected Components)
class UnionFind:
    """
    Union Find data structure for efficiently tracking disjoint sets.
    Uses path compression and union by rank optimization.
    
    Time complexity: 
    - Find: O(α(n)) - amortized constant time, where α is the inverse Ackermann function
    - Union: O(α(n))
    Space complexity: O(n)
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n  # Number of connected components
    
    def find(self, x):
        """Find root of the set with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in the same set
        
        # Attach smaller rank tree under root of higher rank tree
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            # If ranks are same, make one as root and increment its rank
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.count -= 1  # Decrease the count of connected components
        return True

def count_components(n: int, edges: list[list[int]]) -> int:
    """
    Count the number of connected components in an undirected graph.
    
    Time complexity: O(E*α(V)) where E is number of edges and α is inverse Ackermann function
    Space complexity: O(V)
    """
    uf = UnionFind(n)
    
    # Union all connected vertices
    for u, v in edges:
        uf.union(u, v)
    
    return uf.count

# Example 4: Topological Sort (LeetCode 210: Course Schedule II)
def find_order(num_courses: int, prerequisites: list[list[int]]) -> list[int]:
    """
    Return the ordering of courses needed to finish all courses.
    If impossible, return an empty array.
    
    Time complexity: O(V + E)
    Space complexity: O(V + E)
    """
    # Build adjacency list
    graph = defaultdict(list)
    indegree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        indegree[course] += 1
    
    # Initialize queue with courses having no prerequisites
    queue = deque([course for course in range(num_courses) if indegree[course] == 0])
    
    # Result array
    result = []
    
    # Process courses
    while queue:
        current = queue.popleft()
        result.append(current)
        
        # Reduce indegree of neighbors and add to queue if indegree becomes 0
        for neighbor in graph[current]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check if we could finish all courses
    return result if len(result) == num_courses else []

# Example 5: Minimum Spanning Tree - Kruskal's Algorithm
def minimum_spanning_tree(n: int, edges: list[list[int]]) -> int:
    """
    Find the weight of the minimum spanning tree of a connected, undirected graph.
    
    Time complexity: O(E log E) due to sorting
    Space complexity: O(V)
    """
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])
    
    # Initialize Union-Find
    uf = UnionFind(n)
    
    total_weight = 0
    edges_used = 0
    
    # Process edges in order of increasing weight
    for u, v, weight in edges:
        # If including this edge doesn't form a cycle
        if uf.union(u, v):
            total_weight += weight
            edges_used += 1
            
            # MST has V-1 edges for a connected graph with V vertices
            if edges_used == n - 1:
                break
    
    # Check if MST connects all vertices
    return total_weight if edges_used == n - 1 else -1

# Example 6: Floyd-Warshall Algorithm - All Pairs Shortest Path
def floyd_warshall(graph: list[list[int]]) -> list[list[int]]:
    """
    Find shortest paths between all pairs of vertices.
    
    Time complexity: O(V³)
    Space complexity: O(V²)
    """
    n = len(graph)
    dist = [row[:] for row in graph]  # Copy the graph
    
    # Replace non-edges with infinity
    for i in range(n):
        for j in range(n):
            if i != j and dist[i][j] == 0:
                dist[i][j] = float('inf')
    
    # Consider all vertices as intermediates
    for k in range(n):
        # Pick all vertices as source
        for i in range(n):
            # Pick all vertices as destination
            for j in range(n):
                # If vertex k is on the shortest path from i to j,
                # update the value of dist[i][j]
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist

# Example 7: Strongly Connected Components - Kosaraju's Algorithm
def strongly_connected_components(graph: dict) -> list[list[int]]:
    """
    Find all strongly connected components in a directed graph.
    
    Time complexity: O(V + E)
    Space complexity: O(V)
    """
    n = len(graph)
    
    # First DFS to fill the stack
    visited = [False] * n
    stack = []
    
    def fill_order(v):
        visited[v] = True
        for neighbor in graph[v]:
            if not visited[neighbor]:
                fill_order(neighbor)
        stack.append(v)
    
    # Fill order to get vertices in order of finish time
    for i in range(n):
        if not visited[i]:
            fill_order(i)
    
    # Create the transpose graph (reverse all edges)
    transpose = defaultdict(list)
    for i in range(n):
        for j in graph[i]:
            transpose[j].append(i)
    
    # Second DFS to find SCCs
    visited = [False] * n
    sccs = []
    
    def dfs_scc(v, scc):
        visited[v] = True
        scc.append(v)
        for neighbor in transpose[v]:
            if not visited[neighbor]:
                dfs_scc(neighbor, scc)
    
    # Process vertices in order of finish time
    while stack:
        v = stack.pop()
        if not visited[v]:
            scc = []
            dfs_scc(v, scc)
            sccs.append(scc)
    
    return sccs

# Test the examples
if __name__ == "__main__":
    # Example 1: Graph Traversal
    g = Graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    g.add_edge(2, 3)
    g.add_edge(3, 3)
    
    print("DFS Traversal from vertex 2:")
    print(g.dfs(2))  # [2, 0, 1, 3]
    
    print("\nBFS Traversal from vertex 2:")
    print(g.bfs(2))  # [2, 0, 3, 1]
    
    # Example 2: Dijkstra's Algorithm
    times = [[2, 1, 1], [2, 3, 1], [3, 4, 1]]
    n = 4
    k = 2
    print("\nNetwork Delay Time:")
    print(network_delay_time(times, n, k))  # 2
    
    # Example 3: Union Find - Connected Components
    edges = [[0, 1], [1, 2], [3, 4]]
    n = 5
    print("\nNumber of Connected Components:")
    print(count_components(n, edges))  # 2
    
    # Example 4: Topological Sort
    num_courses = 4
    prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]
    print("\nCourse Schedule Order:")
    print(find_order(num_courses, prerequisites))  # [0, 1, 2, 3] or [0, 2, 1, 3]
    
    # Example 5: Minimum Spanning Tree
    edges_mst = [[0, 1, 10], [0, 2, 6], [0, 3, 5], 
                [1, 3, 15], [2, 3, 4]]
    print("\nMinimum Spanning Tree Weight:")
    print(minimum_spanning_tree(4, edges_mst))  # 19
    
    # Example 6: Floyd-Warshall Algorithm
    graph_fw = [
        [0, 5, 0, 10],
        [0, 0, 3, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ]
    print("\nAll Pairs Shortest Paths:")
    result_fw = floyd_warshall(graph_fw)
    for row in result_fw:
        print([float("inf") if x == float("inf") else x for x in row])
    
    # Example 7: Strongly Connected Components
    graph_scc = {
        0: [1],
        1: [2],
        2: [0, 3],
        3: [4],
        4: []
    }
    print("\nStrongly Connected Components:")
    print(strongly_connected_components(graph_scc))  # [[0, 2, 1], [3], [4]] 