def count_paths(graph, start, end):
    path_count = 0  
    def dfs(current, end, visited):
        nonlocal path_count
        if current == end:
            path_count += 1
            return
        visited.add(current)
        for neighbor in graph[current]:
            if neighbor not in visited:
                dfs(neighbor, end, visited)
        visited.remove(current)
    dfs(start, end, set())
    return path_count
graph = {
    'A': ['F'],
    'B': ['G','H'], 
    'C': ['G','I'],
    'D': ['H','I'], 
    'F':['A','B'],
    'G':['C','B'],
    'H':['B','D'],
    'I':['C','D','E'],
    'J':[]
}

start_node = 'A'
end_node = 'J'
path_num = count_paths(graph, start_node, end_node)
print(f"从 {start_node} 到 {end_node} 的路径数量: {path_num}")
