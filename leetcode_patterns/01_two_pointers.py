"""
Two Pointers Pattern

The two pointers pattern uses two pointers to iterate through a data structure. 
These pointers can move independently at different speeds or towards each other.

Common scenarios:
1. Fast and slow pointers (detect cycles in linked list)
2. Left and right pointers (palindrome check, container with most water)
3. Same direction pointers (remove duplicates, find target sum)
"""

# Example 1: Check if string is palindrome (similar to LeetCode 125)
def is_palindrome(s: str) -> bool:
    """
    Using two pointers starting from opposite ends and moving toward the center.
    Time complexity: O(n)
    Space complexity: O(1)
    """
    # Convert to lowercase and keep only alphanumeric characters
    s = ''.join(char.lower() for char in s if char.isalnum())
    
    # Initialize pointers
    left, right = 0, len(s) - 1
    
    # Move pointers inward until they meet
    while left < right:
        # If characters don't match, it's not a palindrome
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    
    # If we've checked all characters without finding a mismatch, it's a palindrome
    return True

# Example 2: Find pair with target sum (similar to LeetCode 167 - Two Sum II)
def two_sum_sorted(nums: list[int], target: int) -> list[int]:
    """
    Using two pointers for sorted array to find pair with target sum.
    Time complexity: O(n)
    Space complexity: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        current_sum = nums[left] + nums[right]
        
        if current_sum == target:
            # Return 1-indexed positions (for LeetCode compatibility)
            return [left + 1, right + 1]
        elif current_sum < target:
            # If sum is too small, increase left pointer to get larger values
            left += 1
        else:
            # If sum is too large, decrease right pointer to get smaller values
            right -= 1
    
    return []  # No solution found

# Example 3: Remove duplicates from sorted array (LeetCode 26)
def remove_duplicates(nums: list[int]) -> int:
    """
    Using two pointers to track unique elements.
    Time complexity: O(n)
    Space complexity: O(1)
    """
    if not nums:
        return 0
    
    # Position where next unique element should be placed
    next_unique = 1
    
    # Iterate through the array starting from the second element
    for i in range(1, len(nums)):
        # If current element is different from the previous one
        if nums[i] != nums[i - 1]:
            # Place it at the next_unique position
            nums[next_unique] = nums[i]
            next_unique += 1
    
    # Return the number of unique elements
    return next_unique

# Example 4: Fast and slow pointers - detect cycle in linked list (LeetCode 141)
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head: ListNode) -> bool:
    """
    Using fast and slow pointers (Floyd's Cycle Finding Algorithm).
    Time complexity: O(n)
    Space complexity: O(1)
    
    This is like a race track - if there's a cycle, the fast runner
    will eventually catch up to the slow runner.
    """
    if not head or not head.next:
        return False
    
    # Initialize slow and fast pointers
    slow = head
    fast = head
    
    # Move slow by 1 step and fast by 2 steps
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        # If they meet, there's a cycle
        if slow == fast:
            return True
    
    # If fast pointer reaches the end, there's no cycle
    return False

# Test the examples
if __name__ == "__main__":
    print("Palindrome Check:")
    print(is_palindrome("A man, a plan, a canal: Panama"))  # True
    print(is_palindrome("race a car"))  # False
    
    print("\nTwo Sum (Sorted Array):")
    print(two_sum_sorted([2, 7, 11, 15], 9))  # [1, 2]
    
    print("\nRemove Duplicates:")
    nums = [1, 1, 2, 2, 3, 4, 4, 5]
    length = remove_duplicates(nums)
    print(f"Length: {length}, Modified array: {nums[:length]}")  # Length: 5, Modified array: [1, 2, 3, 4, 5]
    
    print("\nCycle Detection cannot be tested directly - linked list example") 