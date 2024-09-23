import random

class TreeNode:  
    def __init__(self, key):  
        self.left = None  
        self.right = None  
        self.val = key  
  
def insert(root, key):  
    if root is None:  
        return TreeNode(key)  
    else:  
        if root.val < key:  
            root.right = insert(root.right, key)  
        else:  
            root.left = insert(root.left, key)  
    return root  
  
def inorder_traversal(root):  
    result = []  
    if root:  
        result += inorder_traversal(root.left)  
        result.append(root.val)  
        result += inorder_traversal(root.right)  
    return result  

def generate_random_list(size):
    return [random.randint(1, 1000) for _ in range(size)]

keys = generate_random_list(10)
root = None  
for key in keys:  
    root = insert(root, key)  
  
sorted_keys = inorder_traversal(root)  
print(sorted_keys)  