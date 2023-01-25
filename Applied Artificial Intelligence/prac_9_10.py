# BFS and DFS
import collections


# BFS
def bfs(graph, root):
    visited, queue = set(), collections.deque([root])
    visited.add(root)

    while queue:
        vertex = queue.popleft()
        print(f"{str(vertex)} ", end=" ")

        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)


# DFS
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)

    print(start)

    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited


dfs_graph = {0: set([1, 2]),
             1: set([0, 3, 4]),
             2: set([0]),
             3: set([1]),
             4: set([2, 3])
             }

bfs_graph = {0: [1, 2], 1: [2], 2: [3], 3: [1, 2]}

print("---------- DFS Traversal ------------")
dfs(dfs_graph, 0)

print("----------- BFS Traversal ------------")
bfs(bfs_graph, 0)
