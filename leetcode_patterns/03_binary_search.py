"""
Binary Search Pattern

Binary search is an efficient algorithm for finding a target value in a sorted array.
It works by dividing the search space in half at each step.

Key variations:
1. Classic binary search (find exact target)
2. Modified binary search (first/last occurrence, rotation point)
3. Binary search on answer (find a value that satisfies certain constraints)

Time complexity: O(log n)
Space complexity: O(1) for iterative, O(log n) for recursive due to call stack
"""

# Example 1: Classic Binary Search (LeetCode 704)
def binary_search(nums: list[int], target: int) -> int:
    """
    Classic binary search to find target in a sorted array.
    Returns index of target or -1 if not found.
    
    Time complexity: O(log n)
    Space complexity: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        # Calculate middle index (prevents integer overflow)
        # Alternative: mid = (left + right) // 2
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid  # Target found
        elif nums[mid] < target:
            left = mid + 1  # Search in the right half
        else:
            right = mid - 1  # Search in the left half
    
    return -1  # Target not found

# Example 2: Find First and Last Position (LeetCode 34)
def search_range(nums: list[int], target: int) -> list[int]:
    """
    Find first and last position of target in sorted array.
    Returns [-1, -1] if target is not found.
    
    Time complexity: O(log n)
    Space complexity: O(1)
    """
    # Helper function to find the leftmost (first) occurrence
    def find_left_boundary():
        left, right = 0, len(nums) - 1
        first_pos = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                # Found target, but continue searching left for earlier occurrences
                first_pos = mid
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
                
        return first_pos
    
    # Helper function to find the rightmost (last) occurrence
    def find_right_boundary():
        left, right = 0, len(nums) - 1
        last_pos = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                # Found target, but continue searching right for later occurrences
                last_pos = mid
                left = mid + 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
                
        return last_pos
    
    # Find the boundaries
    left_boundary = find_left_boundary()
    
    # If target not found, return [-1, -1]
    if left_boundary == -1:
        return [-1, -1]
    
    # Find the right boundary
    right_boundary = find_right_boundary()
    
    return [left_boundary, right_boundary]

# Example 3: Search in Rotated Sorted Array (LeetCode 33)
def search_rotated(nums: list[int], target: int) -> int:
    """
    Search in rotated sorted array.
    Returns index of target or -1 if not found.
    
    Time complexity: O(log n)
    Space complexity: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        # Target found
        if nums[mid] == target:
            return mid
        
        # Check if left half is sorted
        if nums[left] <= nums[mid]:
            # Check if target is in the left sorted half
            if nums[left] <= target < nums[mid]:
                right = mid - 1  # Search left half
            else:
                left = mid + 1   # Search right half
        # Right half is sorted
        else:
            # Check if target is in the right sorted half
            if nums[mid] < target <= nums[right]:
                left = mid + 1   # Search right half
            else:
                right = mid - 1  # Search left half
    
    return -1  # Target not found

# Example 4: Binary Search on Answer - Sqrt(x) (LeetCode 69)
def my_sqrt(x: int) -> int:
    """
    Find the square root of x (integer part).
    Uses binary search on the answer space.
    
    Time complexity: O(log x)
    Space complexity: O(1)
    """
    if x == 0:
        return 0
    
    # Define search space: 1 to x
    left, right = 1, x
    
    # Result to track the floor of sqrt(x)
    result = 0
    
    while left <= right:
        mid = left + (right - left) // 2
        
        # Check if mid*mid <= x (prevents integer overflow)
        if mid <= x // mid:
            # mid could be the answer, but we need to find the largest such value
            result = mid
            left = mid + 1
        else:
            # mid is too large
            right = mid - 1
    
    return result

# Test the examples
if __name__ == "__main__":
    print("Classic Binary Search:")
    print(binary_search([1, 2, 3, 4, 5, 6, 7], 5))  # 4
    print(binary_search([1, 2, 3, 4, 5, 6, 7], 8))  # -1
    
    print("\nSearch Range (First and Last Position):")
    print(search_range([5, 7, 7, 8, 8, 10], 8))    # [3, 4]
    print(search_range([5, 7, 7, 8, 8, 10], 6))    # [-1, -1]
    
    print("\nSearch in Rotated Sorted Array:")
    print(search_rotated([4, 5, 6, 7, 0, 1, 2], 0))  # 4
    print(search_rotated([4, 5, 6, 7, 0, 1, 2], 3))  # -1
    
    print("\nSquare Root using Binary Search:")
    print(my_sqrt(4))   # 2
    print(my_sqrt(8))   # 2
    print(my_sqrt(16))  # 4 