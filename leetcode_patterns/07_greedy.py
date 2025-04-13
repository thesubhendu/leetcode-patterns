"""
Greedy Algorithm Pattern

A greedy algorithm makes locally optimal choices at each step with the hope of finding the global optimum.
It's often used for optimization problems where:
1. There is a clear "best choice" at each step
2. Local optimal choices lead to a global optimal solution

Unlike dynamic programming, greedy algorithms don't consider all possible options or revisit decisions.

Key characteristics:
- Simple and straightforward
- Makes decisions based on currently available information
- Never reconsiders previous choices
- May not always produce the optimal solution for all problems

Common applications:
- Interval scheduling
- Coin change (for specific systems)
- Huffman coding
- Minimum spanning trees
- Shortest paths (Dijkstra's algorithm)
"""

# Example 1: Jump Game (LeetCode 55)
def can_jump(nums: list[int]) -> bool:
    """
    Determine if you can reach the last index starting from the first index.
    At each position, you can jump up to nums[i] steps forward.
    
    Time complexity: O(n)
    Space complexity: O(1)
    """
    # Keep track of the furthest position we can reach
    max_reach = 0
    n = len(nums)
    
    # Iterate through the array
    for i in range(n):
        # If we can't reach the current position, return False
        if i > max_reach:
            return False
        
        # Update the furthest position we can reach
        max_reach = max(max_reach, i + nums[i])
        
        # If we can already reach the end, return True
        if max_reach >= n - 1:
            return True
    
    # If we've gone through the entire array without returning, we can reach the end
    return True

# Example 2: Activity Selection (Interval Scheduling)
def max_activities(start: list[int], finish: list[int]) -> list[int]:
    """
    Select the maximum number of non-overlapping activities that can be performed.
    
    Time complexity: O(n log n) - dominated by sorting
    Space complexity: O(n) - for storing result and sorted indices
    """
    n = len(start)
    
    # Create a list of activities with their start and finish times
    activities = [(start[i], finish[i], i) for i in range(n)]
    
    # Sort activities by finish time (key to the greedy approach)
    activities.sort(key=lambda x: x[1])
    
    # Initialize the result with the first activity (finishes earliest)
    result = [activities[0][2]]  # Store original indices
    last_finish = activities[0][1]
    
    # Consider all activities one by one
    for i in range(1, n):
        # If this activity starts after the last selected activity finishes
        if activities[i][0] >= last_finish:
            # Add this activity to the result
            result.append(activities[i][2])
            # Update the last finish time
            last_finish = activities[i][1]
    
    return result

# Example 3: Fractional Knapsack
def fractional_knapsack(weights: list[int], values: list[int], capacity: int) -> float:
    """
    Select items (or fractions of items) to maximize value while not exceeding capacity.
    Unlike 0/1 knapsack, we can take fractions of items.
    
    Time complexity: O(n log n) - dominated by sorting
    Space complexity: O(n) - for storing value-to-weight ratios
    """
    n = len(weights)
    
    # Create a list of value/weight ratios for all items
    ratios = [(values[i] / weights[i], weights[i], values[i]) for i in range(n)]
    
    # Sort items by value-to-weight ratio in decreasing order
    ratios.sort(reverse=True)
    
    total_value = 0
    remaining_capacity = capacity
    
    # Consider items in decreasing order of value-to-weight ratio
    for ratio, weight, value in ratios:
        if remaining_capacity >= weight:
            # Take the whole item
            total_value += value
            remaining_capacity -= weight
        else:
            # Take a fraction of the item
            total_value += ratio * remaining_capacity
            break  # Knapsack is full
    
    return total_value

# Example 4: Huffman Coding
import heapq
from collections import Counter

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    # For priority queue comparison
    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encoding(text: str) -> dict:
    """
    Generate Huffman codes for characters in the text.
    Used for lossless data compression.
    
    Time complexity: O(n log n) where n is the number of unique characters
    Space complexity: O(n)
    """
    if not text:
        return {}
    
    # Count frequency of each character
    freq = Counter(text)
    
    # Create a priority queue (min heap) of Huffman nodes
    priority_queue = [HuffmanNode(char, freq) for char, freq in freq.items()]
    heapq.heapify(priority_queue)
    
    # Build Huffman tree
    while len(priority_queue) > 1:
        # Extract two nodes with lowest frequency
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        
        # Create a new internal node with these two nodes as children
        # The frequency is the sum of the two nodes' frequencies
        internal = HuffmanNode(None, left.freq + right.freq)
        internal.left = left
        internal.right = right
        
        # Add the new node back to the priority queue
        heapq.heappush(priority_queue, internal)
    
    # The remaining node is the root of the Huffman tree
    root = priority_queue[0]
    
    # Generate codes by traversing the tree
    codes = {}
    
    def generate_codes(node, code):
        if node:
            # If it's a leaf node (has a character)
            if node.char:
                codes[node.char] = code
            # Traverse left (add '0')
            generate_codes(node.left, code + '0')
            # Traverse right (add '1')
            generate_codes(node.right, code + '1')
    
    generate_codes(root, '')
    return codes

# Example 5: Minimum Coin Change (for canonical coin systems)
def min_coins_greedy(coins: list[int], amount: int) -> int:
    """
    Find minimum number of coins needed to make a given amount.
    Note: This greedy approach only works for canonical coin systems like US coins (1, 5, 10, 25).
    For general cases, dynamic programming is needed.
    
    Time complexity: O(n log n + amount) where n is the number of coin denominations
    Space complexity: O(1)
    """
    # Sort coins in descending order
    coins.sort(reverse=True)
    
    count = 0
    remaining = amount
    
    # Consider each coin denomination
    for coin in coins:
        # Use as many of this coin as possible
        num_coins = remaining // coin
        count += num_coins
        remaining -= num_coins * coin
        
        # If we've made the full amount, return the count
        if remaining == 0:
            return count
    
    # If we couldn't make the amount with the given coins
    return -1 if remaining > 0 else count

# Example 6: Task Scheduler (LeetCode 621)
def least_interval(tasks: list[str], n: int) -> int:
    """
    Given a list of tasks and a cooling period n, find the least time to finish all tasks.
    Identical tasks must be separated by at least n time units.
    
    Time complexity: O(m) where m is the number of tasks
    Space complexity: O(1) as we only use a fixed-size array of 26 for lowercase letters
    """
    # Count the frequency of each task
    task_counts = Counter(tasks)
    
    # Find the maximum frequency
    max_freq = max(task_counts.values())
    
    # Count how many tasks have the maximum frequency
    max_freq_tasks = sum(1 for count in task_counts.values() if count == max_freq)
    
    # Calculate the result:
    # - We need (max_freq - 1) chunks of (n + 1) time units to handle all but the last occurrence
    #   of the most frequent tasks
    # - We then add max_freq_tasks to account for the last occurrence of each most frequent task
    return max((max_freq - 1) * (n + 1) + max_freq_tasks, len(tasks))

# Test the examples
if __name__ == "__main__":
    print("Jump Game:")
    print(can_jump([2, 3, 1, 1, 4]))  # True
    print(can_jump([3, 2, 1, 0, 4]))  # False
    
    print("\nActivity Selection:")
    start = [1, 3, 0, 5, 8, 5]
    finish = [2, 4, 6, 7, 9, 9]
    print(max_activities(start, finish))  # [0, 1, 3, 4]
    
    print("\nFractional Knapsack:")
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    print(fractional_knapsack(weights, values, capacity))  # 240.0
    
    print("\nHuffman Coding:")
    text = "AAAAABBBCCD"
    codes = huffman_encoding(text)
    print(codes)
    # Example output: {'A': '0', 'B': '10', 'C': '110', 'D': '111'}
    
    print("\nMinimum Coin Change (Greedy):")
    print(min_coins_greedy([1, 5, 10, 25], 63))  # 6 coins (2×25 + 1×10 + 3×1)
    
    print("\nTask Scheduler:")
    print(least_interval(["A", "A", "A", "B", "B", "B"], 2))  # 8 time units 