"""
Sliding Window Pattern

The sliding window pattern involves creating a "window" that can either:
1. Have fixed size (fixed window)
2. Expand or contract based on certain conditions (dynamic window)

This pattern is especially useful for solving array/string problems where we need 
to find subarrays/substrings that satisfy certain conditions.

Time complexity is typically O(n) as we process each element at most twice
(once when it enters the window, once when it exits).
"""

# Example 1: Fixed Window - Find maximum sum subarray of size k (Common sliding window problem)
def max_subarray_sum_fixed(arr: list[int], k: int) -> int:
    """
    Using fixed-size sliding window to find maximum sum of k consecutive elements.
    Time complexity: O(n)
    Space complexity: O(1)
    """
    n = len(arr)
    
    # Edge case: if array is smaller than window size
    if n < k:
        return None
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide the window by removing the first element and adding the next element
    for i in range(k, n):
        # Subtract the element going out of the window
        # Add the element coming into the window
        window_sum = window_sum - arr[i - k] + arr[i]
        
        # Update maximum sum if current window sum is larger
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# Example 2: Dynamic Window - Longest substring with distinct characters (LeetCode 3)
def longest_substring_without_repeating(s: str) -> int:
    """
    Using dynamic-size sliding window to find longest substring without repeating characters.
    Time complexity: O(n)
    Space complexity: O(min(m, n)) where m is size of character set
    """
    n = len(s)
    char_index_map = {}  # Maps character to its most recent index
    max_length = 0
    window_start = 0
    
    # Extend the window
    for window_end in range(n):
        right_char = s[window_end]
        
        # If the character is already in the current window, shrink the window
        if right_char in char_index_map and char_index_map[right_char] >= window_start:
            # Move window_start to the position after the previous occurrence
            window_start = char_index_map[right_char] + 1
        
        # Update the character's most recent index
        char_index_map[right_char] = window_end
        
        # Update the maximum length
        current_length = window_end - window_start + 1
        max_length = max(max_length, current_length)
    
    return max_length

# Example 3: Dynamic Window with target sum - Minimum size subarray sum (LeetCode 209)
def min_subarray_sum(nums: list[int], target: int) -> int:
    """
    Using dynamic-size sliding window to find minimum length subarray with sum >= target.
    Time complexity: O(n)
    Space complexity: O(1)
    """
    n = len(nums)
    window_sum = 0
    min_length = float('inf')
    window_start = 0
    
    for window_end in range(n):
        # Add the next element to the window
        window_sum += nums[window_end]
        
        # Shrink the window as small as possible while maintaining the sum >= target
        while window_sum >= target:
            # Update minimum length
            current_length = window_end - window_start + 1
            min_length = min(min_length, current_length)
            
            # Remove the starting element and shrink the window
            window_sum -= nums[window_start]
            window_start += 1
    
    # Return min_length, or 0 if no subarray found
    return min_length if min_length != float('inf') else 0

# Example 4: Sliding Window with frequency counter - Find all anagrams (LeetCode 438)
from collections import Counter

def find_anagrams(s: str, p: str) -> list[int]:
    """
    Using sliding window with frequency counter to find all anagrams of pattern p in string s.
    Time complexity: O(n)
    Space complexity: O(k) where k is the number of unique characters
    """
    result = []
    n, m = len(s), len(p)
    
    # Edge case: pattern longer than string
    if m > n:
        return result
    
    # Create frequency counters
    p_counter = Counter(p)  # Count characters in pattern
    window_counter = Counter(s[:m])  # Count characters in first window
    
    # Check if first window is an anagram
    if window_counter == p_counter:
        result.append(0)
    
    # Slide the window
    for i in range(m, n):
        # Add the character entering the window
        window_counter[s[i]] += 1
        
        # Remove the character leaving the window
        window_counter[s[i - m]] -= 1
        
        # If count becomes zero, remove the character from counter
        if window_counter[s[i - m]] == 0:
            del window_counter[s[i - m]]
        
        # Check if current window is an anagram
        if window_counter == p_counter:
            result.append(i - m + 1)  # Index of the start of the window
    
    return result

# Test the examples
if __name__ == "__main__":
    print("Max Subarray Sum (Fixed Window):")
    print(max_subarray_sum_fixed([2, 1, 5, 1, 3, 2], 3))  # 9 (subarray [5, 1, 3])
    
    print("\nLongest Substring Without Repeating Characters:")
    print(longest_substring_without_repeating("abcabcbb"))  # 3 ("abc")
    print(longest_substring_without_repeating("bbbbb"))  # 1 ("b")
    
    print("\nMinimum Size Subarray Sum:")
    print(min_subarray_sum([2, 3, 1, 2, 4, 3], 7))  # 2 (subarray [4, 3])
    
    print("\nFind All Anagrams:")
    print(find_anagrams("cbaebabacd", "abc"))  # [0, 6] (anagrams "cba" and "bac") 