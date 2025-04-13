# LeetCode Problem Solving Patterns

This repository contains well-documented examples of common patterns used to solve LeetCode and other coding interview problems. Each pattern is explained with clear examples, detailed code comments, time and space complexity analysis, and reference to specific LeetCode problems.

## Patterns Included

1. **Two Pointers Pattern** (`01_two_pointers.py`)
   - Using multiple pointers to iterate through data structures efficiently
   - Examples: Palindrome check, two sum in sorted array, removing duplicates, cycle detection

2. **Sliding Window Pattern** (`02_sliding_window.py`)
   - Creating fixed or dynamic size "windows" to process consecutive elements 
   - Examples: Maximum sum subarray, longest substring without repeating characters, minimum size subarray, anagram finding

3. **Binary Search Pattern** (`03_binary_search.py`)
   - Efficiently finding elements or answers in sorted arrays or defined ranges
   - Examples: Classic binary search, finding first/last position, search in rotated array, sqrt(x) implementation

4. **Depth-First Search (DFS) & Breadth-First Search (BFS)** (`04_dfs_and_bfs.py`)
   - Graph and tree traversal techniques for exploring connected structures
   - Examples: Tree traversals, level order traversal, number of islands, shortest path in binary matrix, cycle detection

5. **Dynamic Programming** (`05_dynamic_programming.py`)
   - Breaking problems into overlapping subproblems and storing results to avoid redundant calculations
   - Examples: Fibonacci, climbing stairs, coin change, longest common subsequence, knapsack problem, minimum path sum

6. **Backtracking** (`06_backtracking.py`)
   - Incrementally building solutions and abandoning paths that cannot lead to valid solutions
   - Examples: Subsets, permutations, letter combinations, N-Queens, combination sum, palindrome partitioning

7. **Greedy Algorithms** (`07_greedy.py`)
   - Making locally optimal choices at each step to find the global optimum
   - Examples: Jump game, activity selection, fractional knapsack, Huffman coding, task scheduler
  
8. **Graph Algorithms** (`08_graphs.py`)
   - Specialized algorithms for graph problems
   - Examples: Dijkstra's algorithm, Union Find, topological sort, minimum spanning tree, Floyd-Warshall, strongly connected components

## How to Use This Repository

Each pattern file includes:
- An overview of the pattern and when to use it
- Multiple examples showing the pattern in action
- Detailed code comments explaining the implementation
- Time and space complexity analysis
- Test cases to run and verify the solutions

To understand a specific pattern:
1. Read the pattern overview comments at the top of each file
2. Study the examples and their implementations
3. Run the file to see the examples in action
4. Try to solve similar LeetCode problems using the pattern

## Running the Examples

Each file can be run independently to see the examples in action:

```bash
python 01_two_pointers.py
python 02_sliding_window.py
# etc.
```

## Additional Resources

- [LeetCode](https://leetcode.com/) - Practice more problems
- [Cracking the Coding Interview](http://www.crackingthecodinginterview.com/) - Book with many pattern explanations
- [Grokking the Coding Interview](https://www.educative.io/courses/grokking-the-coding-interview) - Course organized by patterns 