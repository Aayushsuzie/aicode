# 4 Write a python program to implement uniform cost search.

# import heapq
# from collections import defaultdict

# def uniform_cost_search(graph, start, goal):
#     priority_queue = [(0, start)]
#     cost_so_far = {start: 0}
#     parent = {start: None}
#     visited = set()
    
#     while priority_queue:
#         current_cost, current_node = heapq.heappop(priority_queue)
        
#         if current_node in visited:
#             continue
        
#         visited.add(current_node)
        
#         if current_node == goal:
#             path = []
#             while current_node is not None:
#                 path.append(current_node)
#                 current_node = parent[current_node]
#             path.reverse()
#             return path, current_cost
        
#         for neighbor, cost in graph[current_node].items():
#             new_cost = current_cost + cost
            
#             if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
#                 cost_so_far[neighbor] = new_cost
#                 parent[neighbor] = current_node
#                 heapq.heappush(priority_queue, (new_cost, neighbor))
    
#     return None, float('inf')

# if __name__ == "__main__":
#     graph = defaultdict(dict, {
#         'A': {'B': 1, 'C': 4},
#         'B': {'A': 1, 'D': 2, 'E': 5},
#         'C': {'A': 4, 'F': 3},
#         'D': {'B': 2, 'G': 2},
#         'E': {'B': 5, 'G': 2},
#         'F': {'C': 3, 'G': 6},
#         'G': {'D': 1, 'E': 2, 'F': 6}
#     })
    
#     start_node = 'A'
#     goal_node = 'G'
    
#     path, cost = uniform_cost_search(graph, start_node, goal_node)
    
#     if path:
#         print(f"Path found: {' -> '.join(path)}\nTotal cost : {cost}")
#     else:
#         print("No path found.")


# 5. Write a python program to implement Greedy best first search.

# import heapq

# def greedy_best_first_search(graph, heuristic, start, goal):
#     priority_queue = [(heuristic[start], 0, start)]  # (heuristic, path_cost, node)
#     parent = {start: None}
#     cost_so_far = {start: 0}
#     visited = set()
    
#     while priority_queue:
#         current_heuristic, current_cost, current_node = heapq.heappop(priority_queue)
        
#         if current_node in visited:
#             continue
        
#         visited.add(current_node)
        
#         # Goal check
#         if current_node == goal:
#             # Reconstruct path
#             path = []
#             while current_node:
#                 path.append(current_node)
#                 current_node = parent[current_node]
#             return path[::-1], current_cost  # Reverse path and return total cost

#         # Explore neighbors
#         for neighbor, edge_cost in graph[current_node].items():
#             new_cost = current_cost + edge_cost
#             if neighbor not in visited or new_cost < cost_so_far.get(neighbor, float('inf')):
#                 parent[neighbor] = current_node
#                 cost_so_far[neighbor] = new_cost
#                 heapq.heappush(priority_queue, (heuristic[neighbor], new_cost, neighbor))
    
#     return None, float('inf')  # No path found

# if __name__ == "__main__":
#     # Example graph with weights
#     graph = {
#         'A': {'B': 1, 'C': 4},
#         'B': {'D': 2, 'E': 5},
#         'C': {'F': 3},
#         'D': {'G': 6},
#         'E': {'G': 2},
#         'F': {'G': 4},
#         'G': {}
#     }
    
#     # Heuristic function estimating cost to reach goal 'G' from each node
#     heuristic = {
#         'A': 7,
#         'B': 6,
#         'C': 2,
#         'D': 4,
#         'E': 1,
#         'F': 3, 
#         'G': 0
#     }
    
#     start_node = 'A'
#     goal_node = 'G'
    
#     path, total_cost = greedy_best_first_search(graph, heuristic, start_node, goal_node)
    
#     if path:
#         print(f"Path found: {' -> '.join(path)}\nTotal path cost: {total_cost}")
#     else:
#         print("No path found.")


#6. Write a python program to implement A* Search.

# import heapq

# def a_star_search(graph, heuristic, start, goal):
#     priority_queue = [(0 + heuristic[start], 0, start)]  # (f(n), g(n), node)
#     parent = {start: None}
#     cost_so_far = {start: 0}
    
#     while priority_queue:
#         current_f, current_cost, current_node = heapq.heappop(priority_queue)
        
#         if current_node == goal:
#             path = []
#             while current_node is not None:
#                 path.append(current_node)
#                 current_node = parent[current_node]
#             path.reverse()
#             return path, current_cost
        
#         for neighbor, edge_cost in graph[current_node].items():
#             new_cost = current_cost + edge_cost
#             f_cost = new_cost + heuristic[neighbor]
            
#             if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
#                 cost_so_far[neighbor] = new_cost
#                 parent[neighbor] = current_node
#                 heapq.heappush(priority_queue, (f_cost, new_cost, neighbor))
    
#     return None, float('inf')

# if __name__ == "__main__":
#     # Example graph with weights
#     graph = {
#         'A': {'B': 1, 'C': 4},
#         'B': {'D': 2, 'E': 5},
#         'C': {'F': 3},
#         'D': {'G': 6},
#         'E': {'G': 2},
#         'F': {'G': 4},
#         'G': {}
#     }
    
#     # Heuristic function estimating cost to reach goal 'G' from each node
#     heuristic = {
#         'A': 7,
#         'B': 6,
#         'C': 2,
#         'D': 4,
#         'E': 1,
#         'F': 3,
#         'G': 0
#     }
    
#     start_node = 'A'
#     goal_node = 'G'
    
#     path, cost = a_star_search(graph, heuristic, start_node, goal_node)
    
#     if path:
#         print(f"Path found: {' -> '.join(path)}\nTotal cost: {cost}")
#     else:
#         print("No path found.")

#7. Write a python program to find the maximum score that the maximizing player can get using Minimax Algorithm.

def minimax(node, is_maximizing_player):

    if not node['children']:  # If no children, it's a leaf node
        return node['value']
    
    if is_maximizing_player:
        best_value = float('-inf')
        for child in node['children']:
            value = minimax(child, False)  # Call minimax for the minimizing player
            best_value = max(best_value, value)  # Choose the maximum value
        return best_value
    else:
        best_value = float('inf')
        for child in node['children']:
            value = minimax(child, True)  # Call minimax for the maximizing player
            best_value = min(best_value, value)  # Choose the minimum value
        return best_value


game_tree = {
    'value': None, 
    'children': [
        {
            'value': None,
            'children': [
                {'value': 14, 'children': []},  # Leaf node
                {'value': 6, 'children': []},   # Leaf node
                {'value': 8, 'children': []}    # Leaf node
            ]
        },
        {
            'value': None,
            'children': [
                {'value': 7, 'children': []},   # Leaf node
                {'value': 3, 'children': []},   # Leaf node
                {'value': 15, 'children': []}   # Leaf node
            ]
        },
        {
            'value': None,
            'children': [
                {'value': 20, 'children': []},  # Leaf node
                {'value': 5, 'children': []},   # Leaf node
                {'value': 12, 'children': []}   # Leaf node
            ]
        }
    ]
}

max_score = minimax(game_tree, True) 

print(f"The maximum score that the maximizing player can get is: {max_score}")
