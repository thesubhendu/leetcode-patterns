class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        #find length of both string
        l1 = len(word1)
        l2 = len(word2)
        st = ''
        pointer = 0

        while(pointer < l1 or pointer < l2):
            if(pointer < l1):
                st += word1[pointer]

            if(pointer < l2):
                st += word2[pointer]
            pointer+=1

        return st


solution = Solution()
result = solution.mergeAlternately("abc", "123")
print(result)  # This will print "a1b2c3"