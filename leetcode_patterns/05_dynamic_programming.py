"""
Dynamic Programming Pattern

Dynamic Programming (DP) is a technique for solving problems by breaking them down
into overlapping subproblems and storing the results to avoid redundant calculations.

Key characteristics:
1. Optimal substructure: Optimal solution can be constructed from optimal solutions of subproblems
2. Overlapping subproblems: Same subproblems recur and can be cached

Two main approaches:
1. Top-down with memoization (recursive with caching)
2. Bottom-up with tabulation (iterative)

Common DP patterns:
- 1D DP (sequences, decision-making)
- 2D DP (grid problems, string problems)
- State machine DP
- Interval DP
"""

# Example 1: Fibonacci Sequence - Classic DP Example
def fibonacci():
    """
    Classic example showing the power of DP.
    Time complexity without DP: O(2^n) - exponential
    Time complexity with DP: O(n) - linear
    """
    n = 40  # Try computing the 40th Fibonacci number
    
    # Naive recursive approach (very slow for large n)
    def fib_recursive(n):
        if n <= 1:
            return n
        return fib_recursive(n-1) + fib_recursive(n-2)
    
    # Top-down DP with memoization
    def fib_memoization(n, memo={}):
        if n in memo:
            return memo[n]
        if n <= 1:
            return n
        memo[n] = fib_memoization(n-1, memo) + fib_memoization(n-2, memo)
        return memo[n]
    
    # Bottom-up DP with tabulation
    def fib_tabulation(n):
        if n <= 1:
            return n
        
        # Initialize DP table
        dp = [0] * (n + 1)
        dp[1] = 1
        
        # Fill table bottom-up
        for i in range(2, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
        
        return dp[n]
    
    # Space-optimized bottom-up approach
    def fib_optimized(n):
        if n <= 1:
            return n
        
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        
        return b
    
    # Computing with different methods
    # Note: fib_recursive(40) would take too long to compute
    
    # Only run recursive for small values to demonstrate exponential time
    recursive_result = fib_recursive(10) if n > 10 else fib_recursive(n)
    memoization_result = fib_memoization(n)
    tabulation_result = fib_tabulation(n)
    optimized_result = fib_optimized(n)
    
    return {
        "recursive_result": recursive_result,
        "memoization_result": memoization_result,
        "tabulation_result": tabulation_result,
        "optimized_result": optimized_result
    }

# Example 2: Climbing Stairs (LeetCode 70)
def climb_stairs(n: int) -> int:
    """
    You can climb 1 or 2 steps at a time. How many distinct ways can you climb to the top?
    This is essentially Fibonacci sequence with different initial values.
    
    Time complexity: O(n)
    Space complexity: O(1)
    """
    if n <= 2:
        return n
    
    # Initialize first two steps
    one_step_before = 2  # Ways to climb 2 stairs
    two_steps_before = 1  # Ways to climb 1 stair
    
    # Calculate ways for each step
    for i in range(3, n + 1):
        current = one_step_before + two_steps_before
        two_steps_before = one_step_before
        one_step_before = current
    
    return one_step_before

# Example 3: Coin Change (LeetCode 322) - Unbounded Knapsack Pattern
def coin_change(coins: list[int], amount: int) -> int:
    """
    Find the fewest number of coins needed to make up the given amount.
    
    Time complexity: O(amount * len(coins))
    Space complexity: O(amount)
    """
    # Initialize DP array with amount+1 (which is greater than any possible result)
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # Base case: 0 coins needed to make amount 0
    
    # Fill dp array
    for coin in coins:
        for x in range(coin, amount + 1):
            # Minimum of not using this coin vs. using this coin + minimum for remaining amount
            dp[x] = min(dp[x], dp[x - coin] + 1)
    
    # Return result if possible, otherwise -1
    return dp[amount] if dp[amount] != float('inf') else -1

# Example 4: Longest Common Subsequence (LeetCode 1143) - String/Sequence DP
def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Find the length of the longest subsequence common to both strings.
    
    Time complexity: O(m*n)
    Space complexity: O(m*n)
    """
    m, n = len(text1), len(text2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                # Characters match, add 1 to result from subproblems without these characters
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                # Characters don't match, take maximum from subproblems
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# Example 5: 0/1 Knapsack Problem - Classic DP Problem
def knapsack(weights: list[int], values: list[int], capacity: int) -> int:
    """
    Select items to maximize value while keeping weight under capacity.
    Each item can only be selected once (0/1 property).
    
    Time complexity: O(n*capacity)
    Space complexity: O(n*capacity)
    """
    n = len(weights)
    
    # Create DP table
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                # Can include this item, decide whether to take it or not
                dp[i][w] = max(
                    values[i-1] + dp[i-1][w-weights[i-1]],  # Take item
                    dp[i-1][w]  # Don't take item
                )
            else:
                # Can't include this item (too heavy)
                dp[i][w] = dp[i-1][w]
    
    # Return maximum value
    return dp[n][capacity]

# Example 6: Minimum Path Sum (LeetCode 64) - Grid DP
def min_path_sum(grid: list[list[int]]) -> int:
    """
    Find the path with minimum sum from top-left to bottom-right in a grid.
    Can only move down or right.
    
    Time complexity: O(m*n)
    Space complexity: O(m*n) or O(n) with space optimization
    """
    if not grid or not grid[0]:
        return 0
    
    m, n = len(grid), len(grid[0])
    
    # Create DP table with the same dimensions as grid
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]  # Start with top-left cell
    
    # Fill first row (can only come from left)
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    
    # Fill first column (can only come from above)
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    
    # Fill rest of the DP table
    for i in range(1, m):
        for j in range(1, n):
            # Choose minimum path from above or left
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
    
    # Return minimum path sum to bottom-right
    return dp[m-1][n-1]

# Example 7: House Robber (LeetCode 198) - State Machine DP
def rob(nums: list[int]) -> int:
    """
    Rob houses to maximize amount of money, but can't rob adjacent houses.
    
    Time complexity: O(n)
    Space complexity: O(1)
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    # Only need to keep track of two states:
    # - Max money if we rob current house
    # - Max money if we don't rob current house
    
    rob_current = nums[0]  # Max money if we rob the first house
    skip_current = 0       # Max money if we skip the first house
    
    for i in range(1, len(nums)):
        # If we rob this house, we must have skipped the previous one
        # If we skip this house, take max of rob_current and skip_current
        rob_current, skip_current = skip_current + nums[i], max(rob_current, skip_current)
    
    # Return the maximum amount
    return max(rob_current, skip_current)

# Test the examples
if __name__ == "__main__":
    print("Fibonacci Results:")
    fib_results = fibonacci()
    for method, result in fib_results.items():
        print(f"  {method}: {result}")
    
    print("\nClimbing Stairs:")
    print(climb_stairs(5))  # 8 ways to climb 5 stairs
    
    print("\nCoin Change:")
    print(coin_change([1, 2, 5], 11))  # 3 coins (5 + 5 + 1)
    
    print("\nLongest Common Subsequence:")
    print(longest_common_subsequence("abcde", "ace"))  # 3 ("ace")
    
    print("\n0/1 Knapsack:")
    print(knapsack([1, 2, 3, 5], [1, 6, 10, 16], 7))  # 22 (items with weights 2, 5)
    
    print("\nMinimum Path Sum:")
    print(min_path_sum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]))  # 7 (path: 1→3→1→1→1)
    
    print("\nHouse Robber:")
    print(rob([1, 2, 3, 1]))  # 4 (rob house 1 and 3)
    print(rob([2, 7, 9, 3, 1]))  # 12 (rob house 1, 3, and 5) 