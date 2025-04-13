"""
Backtracking Pattern

Backtracking is an algorithmic technique for finding all (or some) solutions to problems 
by incrementally building candidates and abandoning a candidate as soon as it's determined 
that it cannot lead to a valid solution (pruning).

Key characteristics:
1. Decision making: At each step, make a choice from available options
2. Constraints: Check if current state violates any constraints
3. Goal check: Determine if current state is a solution
4. Pruning: Abandon paths that cannot lead to valid solutions

Common patterns in backtracking:
- Permutations and combinations
- Subset problems
- String partitioning
- Grid/maze traversal
- N-Queens problem
"""

# Example 1: Subsets (LeetCode 78)
def subsets(nums: list[int]) -> list[list[int]]:
    """
    Generate all possible subsets of a given array.
    
    Time complexity: O(N * 2^N) - Generate 2^N subsets, each could require O(N) time to copy
    Space complexity: O(N * 2^N) for storing all subsets
    """
    result = []
    
    def backtrack(start, current):
        # Add the current subset to the result (make a copy)
        result.append(current[:])
        
        # Try each number that hasn't been included yet
        for i in range(start, len(nums)):
            # Include nums[i] in the current subset
            current.append(nums[i])
            
            # Explore further with this number included
            backtrack(i + 1, current)
            
            # Backtrack: remove nums[i] to try the next possibility
            current.pop()
    
    # Start backtracking with empty subset
    backtrack(0, [])
    return result

# Example 2: Permutations (LeetCode 46)
def permutations(nums: list[int]) -> list[list[int]]:
    """
    Generate all possible permutations of a given array.
    
    Time complexity: O(N * N!) - Generate N! permutations, each requires O(N) time to copy
    Space complexity: O(N * N!) for storing all permutations
    """
    result = []
    
    def backtrack(current):
        # Base case: if the permutation is complete (all elements used)
        if len(current) == len(nums):
            result.append(current[:])  # Add a copy of the permutation to results
            return
        
        # Try each number that hasn't been used yet
        for num in nums:
            if num in current:  # Skip if already used in this permutation
                continue
                
            # Include this number in the current permutation
            current.append(num)
            
            # Explore further permutations with this number included
            backtrack(current)
            
            # Backtrack: remove the number to try next possibility
            current.pop()
    
    # Start backtracking with empty permutation
    backtrack([])
    return result

# Example 3: Letter Combinations of a Phone Number (LeetCode 17)
def letter_combinations(digits: str) -> list[str]:
    """
    Given a string containing digits from 2-9, return all possible letter combinations.
    
    Time complexity: O(4^N * N) where N is the number of digits (worst case each digit maps to 4 letters)
    Space complexity: O(N) for recursion stack, O(4^N) for output
    """
    if not digits:
        return []
    
    # Mapping of digits to letters (like on a phone keypad)
    phone_map = {
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz'
    }
    
    result = []
    
    def backtrack(index, current):
        # Base case: if we've processed all digits
        if index == len(digits):
            result.append(''.join(current))
            return
        
        # Get the letters corresponding to the current digit
        for letter in phone_map[digits[index]]:
            # Add the letter to our combination
            current.append(letter)
            
            # Move to the next digit
            backtrack(index + 1, current)
            
            # Backtrack: remove the letter to try next possibility
            current.pop()
    
    # Start backtracking from first digit with empty combination
    backtrack(0, [])
    return result

# Example 4: N-Queens (LeetCode 51)
def solve_n_queens(n: int) -> list[list[str]]:
    """
    Place N queens on an N×N chessboard so that no two queens attack each other.
    
    Time complexity: O(N!) - approximately since we're trying different placements
    Space complexity: O(N) for the board representation
    """
    result = []
    
    # Helper function to check if a position is valid for placing a queen
    def is_valid(board, row, col):
        # Check column (no need to check rows as we place one queen per row)
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check upper-left diagonal
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        
        # Check upper-right diagonal
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1
        
        return True
    
    def backtrack(row, board):
        # Base case: if all queens are placed
        if row == n:
            # Convert the board to the required format
            result.append([''.join(row) for row in board])
            return
        
        # Try placing queen in each column of the current row
        for col in range(n):
            if is_valid(board, row, col):
                # Place queen
                board[row][col] = 'Q'
                
                # Move to the next row
                backtrack(row + 1, board)
                
                # Backtrack: remove the queen to try next possibility
                board[row][col] = '.'
    
    # Initialize the board with empty cells
    board = [['.' for _ in range(n)] for _ in range(n)]
    
    # Start backtracking from the first row
    backtrack(0, board)
    return result

# Example 5: Combination Sum (LeetCode 39)
def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    """
    Find all unique combinations of candidates where the sum equals target.
    Each number can be used multiple times.
    
    Time complexity: O(N^(T/M)) where T is target, M is minimum candidate value, N is number of candidates
    Space complexity: O(T/M) for recursion depth
    """
    result = []
    
    def backtrack(start, current, remaining):
        # Base case: if the target is reached
        if remaining == 0:
            result.append(current[:])
            return
        
        # Base case: if target is overshot
        if remaining < 0:
            return
        
        # Try each candidate from the current position
        for i in range(start, len(candidates)):
            # Include this candidate
            current.append(candidates[i])
            
            # Since we can reuse the same element, we pass i instead of i+1
            backtrack(i, current, remaining - candidates[i])
            
            # Backtrack: remove the element
            current.pop()
    
    # Sort candidates to optimize (allows earlier pruning)
    candidates.sort()
    
    # Start backtracking
    backtrack(0, [], target)
    return result

# Example 6: Palindrome Partitioning (LeetCode 131)
def partition(s: str) -> list[list[str]]:
    """
    Partition string s such that every substring is a palindrome.
    Return all possible palindrome partitioning.
    
    Time complexity: O(N * 2^N) - potentially exponential partitions, each needs palindrome check
    Space complexity: O(N) for recursion depth
    """
    result = []
    
    # Helper function to check if a string is a palindrome
    def is_palindrome(substr):
        return substr == substr[::-1]
    
    def backtrack(start, current):
        # Base case: if we've processed the entire string
        if start == len(s):
            result.append(current[:])
            return
        
        # Try all possible substrings starting from 'start'
        for end in range(start + 1, len(s) + 1):
            # Get the substring
            substring = s[start:end]
            
            # If it's a palindrome, add it to current partition and continue
            if is_palindrome(substring):
                current.append(substring)
                backtrack(end, current)
                current.pop()  # Backtrack
    
    # Start backtracking from the beginning with empty partition
    backtrack(0, [])
    return result

# Test the examples
if __name__ == "__main__":
    print("Subsets:")
    print(subsets([1, 2, 3]))
    # Output: [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
    
    print("\nPermutations:")
    print(permutations([1, 2, 3]))
    # Output: [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    
    print("\nLetter Combinations of a Phone Number:")
    print(letter_combinations("23"))
    # Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]
    
    print("\nN-Queens (N=4):")
    solutions = solve_n_queens(4)
    for solution in solutions:
        for row in solution:
            print(row)
        print()
    # Output: Two solutions for 4-Queens
    
    print("\nCombination Sum:")
    print(combination_sum([2, 3, 6, 7], 7))
    # Output: [[2, 2, 3], [7]]
    
    print("\nPalindrome Partitioning:")
    print(partition("aab"))
    # Output: [["a", "a", "b"], ["aa", "b"]] 