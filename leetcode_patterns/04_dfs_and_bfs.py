"""
Depth-First Search (DFS) and Breadth-First Search (BFS) Patterns

DFS and BFS are fundamental graph traversal algorithms applicable to:
- Graphs (directed, undirected, weighted)
- Trees (binary trees, n-ary trees)
- Matrices (grid problems)

DFS: Explores as far as possible along each branch before backtracking.
    - Uses stack (explicit or implicit via recursion)
    - Good for: Path finding, cycle detection, topological sorting

BFS: Explores all neighbors at the current depth before moving to nodes at the next depth.
    - Uses queue
    - Good for: Shortest path (unweighted), level-order traversal, connected components
"""

# Tree node definition (used in multiple examples)
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Graph representation (used in multiple examples)
class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adj_list = [[] for _ in range(vertices)]
        
    def add_edge(self, v, w):
        self.adj_list[v].append(w)

# Example 1: DFS - Tree Traversal (LeetCode 94, 144, 145)
def tree_dfs_traversal():
    """
    Three common ways to traverse a binary tree using DFS:
    1. Preorder (root, left, right)
    2. Inorder (left, root, right)
    3. Postorder (left, right, root)
    
    Time complexity: O(n)
    Space complexity: O(h) where h is the height of the tree (worst case O(n))
    """
    # Create a sample tree:
    #        1
    #       / \
    #      2   3
    #     / \
    #    4   5
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    
    # Recursive DFS implementations
    def preorder(node, result):
        if not node:
            return
        # Visit root, then left, then right
        result.append(node.val)
        preorder(node.left, result)
        preorder(node.right, result)
    
    def inorder(node, result):
        if not node:
            return
        # Visit left, then root, then right
        inorder(node.left, result)
        result.append(node.val)
        inorder(node.right, result)
    
    def postorder(node, result):
        if not node:
            return
        # Visit left, then right, then root
        postorder(node.left, result)
        postorder(node.right, result)
        result.append(node.val)
    
    # Iterative DFS implementation (preorder)
    def preorder_iterative(root):
        if not root:
            return []
        
        result = []
        stack = [root]  # Use explicit stack
        
        while stack:
            node = stack.pop()  # Pop from the end (LIFO)
            result.append(node.val)
            
            # Push right first so left is processed first (LIFO)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
                
        return result
    
    # Execute and return the results
    preorder_result = []
    preorder(root, preorder_result)
    
    inorder_result = []
    inorder(root, inorder_result)
    
    postorder_result = []
    postorder(root, postorder_result)
    
    preorder_iter_result = preorder_iterative(root)
    
    return {
        "preorder_recursive": preorder_result,      # [1, 2, 4, 5, 3]
        "inorder_recursive": inorder_result,        # [4, 2, 5, 1, 3]
        "postorder_recursive": postorder_result,    # [4, 5, 2, 3, 1]
        "preorder_iterative": preorder_iter_result  # [1, 2, 4, 5, 3]
    }

# Example 2: BFS - Level Order Traversal (LeetCode 102)
from collections import deque

def level_order_traversal(root: TreeNode) -> list[list[int]]:
    """
    BFS traversal of a binary tree to get nodes level by level.
    
    Time complexity: O(n)
    Space complexity: O(w) where w is the max width of the tree (worst case O(n/2))
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])  # Initialize queue with root
    
    while queue:
        level_size = len(queue)  # Number of nodes at current level
        current_level = []
        
        # Process all nodes at the current level
        for _ in range(level_size):
            node = queue.popleft()  # Dequeue node (FIFO)
            current_level.append(node.val)
            
            # Enqueue children for the next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)  # Add current level to result
    
    return result

# Example 3: DFS - Number of Islands (LeetCode 200)
def num_islands(grid: list[list[str]]) -> int:
    """
    Use DFS to count the number of islands in a 2D grid.
    An island is a group of '1's connected horizontally or vertically.
    
    Time complexity: O(m*n) where m is rows and n is columns
    Space complexity: O(m*n) in worst case due to recursion stack
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    # DFS to mark all connected '1's as visited
    def dfs(i, j):
        # Check if out of bounds or if cell is water/visited
        if (i < 0 or i >= rows or 
            j < 0 or j >= cols or 
            grid[i][j] != '1'):
            return
        
        # Mark as visited by changing '1' to '0' or another marker
        grid[i][j] = '0'
        
        # Explore all 4 directions
        dfs(i + 1, j)  # Down
        dfs(i - 1, j)  # Up
        dfs(i, j + 1)  # Right
        dfs(i, j - 1)  # Left
    
    # Iterate through each cell in the grid
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                count += 1    # Found a new island
                dfs(i, j)     # Mark the entire island as visited
    
    return count

# Example 4: BFS - Shortest Path in Binary Matrix (LeetCode 1091)
def shortest_path_binary_matrix(grid: list[list[int]]) -> int:
    """
    Find the shortest path from top-left to bottom-right in a binary matrix.
    A path can only pass through cells containing 0, and you can move in 8 directions.
    
    Time complexity: O(n²) for an n x n grid
    Space complexity: O(n²) for the queue in worst case
    """
    n = len(grid)
    
    # Check if start or end is blocked
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1
    
    # Possible directions: horizontal, vertical, and diagonal
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    # BFS
    queue = deque([(0, 0, 1)])  # (row, col, distance)
    grid[0][0] = 1  # Mark as visited
    
    while queue:
        row, col, distance = queue.popleft()
        
        # If reached the bottom-right corner
        if row == n - 1 and col == n - 1:
            return distance
        
        # Explore all 8 directions
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check if valid and unvisited
            if (0 <= new_row < n and 
                0 <= new_col < n and 
                grid[new_row][new_col] == 0):
                
                grid[new_row][new_col] = 1  # Mark as visited
                queue.append((new_row, new_col, distance + 1))
    
    return -1  # No path found

# Example 5: DFS - Course Schedule (LeetCode 207) - Cycle Detection
def can_finish(num_courses: int, prerequisites: list[list[int]]) -> bool:
    """
    Determine if it's possible to finish all courses given prerequisites.
    This is a graph cycle detection problem.
    
    Time complexity: O(V + E) where V is vertices and E is edges
    Space complexity: O(V + E) for the adjacency list and visited sets
    """
    # Build adjacency list
    graph = [[] for _ in range(num_courses)]
    for course, prereq in prerequisites:
        graph[course].append(prereq)
    
    # Three states for DFS:
    # 0: unvisited
    # 1: visiting (in current DFS path)
    # 2: visited (all descendants processed)
    visited = [0] * num_courses
    
    def dfs(course):
        # If this node is being visited in current DFS path, cycle detected
        if visited[course] == 1:
            return False
        
        # If already fully processed, no need to re-process
        if visited[course] == 2:
            return True
        
        # Mark as being visited in current path
        visited[course] = 1
        
        # Visit all prerequisites
        for prereq in graph[course]:
            if not dfs(prereq):
                return False
        
        # Mark as fully processed
        visited[course] = 2
        return True
    
    # Try to visit all courses
    for course in range(num_courses):
        if not dfs(course):
            return False
    
    return True

# Test the examples
if __name__ == "__main__":
    print("Tree DFS Traversals:")
    traversal_results = tree_dfs_traversal()
    for name, result in traversal_results.items():
        print(f"  {name}: {result}")
    
    # Create a tree for BFS testing
    sample_tree = TreeNode(3)
    sample_tree.left = TreeNode(9)
    sample_tree.right = TreeNode(20)
    sample_tree.right.left = TreeNode(15)
    sample_tree.right.right = TreeNode(7)
    
    print("\nTree BFS Level Order Traversal:")
    print(level_order_traversal(sample_tree))  # [[3], [9, 20], [15, 7]]
    
    print("\nNumber of Islands (DFS):")
    grid1 = [
        ["1", "1", "1", "1", "0"],
        ["1", "1", "0", "1", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "0", "0", "0"]
    ]
    print(num_islands(grid1))  # 1
    
    grid2 = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"]
    ]
    print(num_islands(grid2))  # 3
    
    print("\nShortest Path in Binary Matrix (BFS):")
    binary_matrix = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0]
    ]
    print(shortest_path_binary_matrix(binary_matrix))  # 9
    
    print("\nCourse Schedule - Cycle Detection (DFS):")
    prerequisites1 = [[1, 0]]  # Can finish
    prerequisites2 = [[1, 0], [0, 1]]  # Cannot finish (cycle)
    print(can_finish(2, prerequisites1))  # True
    print(can_finish(2, prerequisites2))  # False 