class Solution:
    def isPalindrome(self, s: str) -> bool:
        # agadi pachadi bata same read hune
            # sanitize input
            # go from both side and see if characters are same
            n = len(s)
            i = 0
            j = n - 1
            
            while i < j:
                # Skip non-alphanumeric characters from left
                while i < j and not s[i].isalnum():
                    i += 1
                # Skip non-alphanumeric characters from right    
                while i < j and not s[j].isalnum():
                    j -= 1
                    
                # Compare characters (case-insensitive)
                if s[i].lower() != s[j].lower():
                    return False
                    
                i += 1
                j -= 1
            
            return True

    
    def subPalindrome(self, s: str) -> bool:
        lengthOfStr = len(s)

        i = 0
        j = lengthOfStr-1

        while (i<j):
            #skip if not alphanumeric
            if not s[i].isalnum():
                i+=1
                continue

            if not s[j].isalnum():
                j-=1
                continue

            if(s[i]!= s[j]):
                return False
            else:
                i+=1
                j-=1

        return True
        