from heapq import heappush, heappop  # Recommended.
import numpy as np

from flightsim.world import World

from .occupancy_map import OccupancyMap # Recommended.

class Node:
    def __init__(self, index, parent, cost, heuristic, metric_coord):
        self.index = index
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

        # This is required as start and goal may not be in the center of the voxel
        self.metric_coord = metric_coord


    def __lt__(self, other):
        return self.cost + self.heuristic < other.cost + other.heuristic

    def __eq__(self, other):
        for i in range(3):
            if self.index[i] != other.index[i]:
                return False
        return True
    
    def __hash__(self):
        return hash((self.index[0], self.index[1], self.index[2]))
    
    def __repr__(self):
        return f"Node({self.index}, {self.parent}, {self.cost}, {self.heuristic})"


def cost_fxn(node, neighbor):
    return node.cost + np.linalg.norm(node.metric_coord - neighbor.metric_coord)


def heuristic_fxn(node, goal_node, astar):
    if not astar:
        return 0
    return np.linalg.norm(node.metric_coord - goal_node.metric_coord)


def get_neighbors(node, occ_map, goal_node, astar):
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if i == 0 and j == 0 and k == 0:
                    continue
                index = (node.index[0] + i, node.index[1] + j, node.index[2] + k)
                if occ_map.is_valid_index(index) and not occ_map.is_occupied_index(index):
                    metric_coord = occ_map.index_to_metric_center(index)
                    if index == goal_node.index:
                        metric_coord = goal_node.metric_coord
                    neighbor_node = Node(index, None, 0, 0, metric_coord)
                    cost = cost_fxn(node, neighbor_node)
                    heuristic_value = heuristic_fxn(neighbor_node, goal_node, astar)
                    neighbor_node.cost = cost
                    neighbor_node.heuristic = heuristic_value
                    neighbors.append(neighbor_node)
                    
    return neighbors


def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))
    start_node = Node(start_index, None, 0, np.linalg.norm(start - goal), start)
    goal_node = Node(goal_index, None, np.inf, 0, goal)

    priority_queue = []
    heappush(priority_queue, start_node)
    cost_map = {start_node: 0}
    nodes_expanded = 0

    while priority_queue:
        current_node = heappop(priority_queue)

        if current_node == goal_node:
            path = []
            while current_node:
                path.append(current_node.metric_coord)
                current_node = current_node.parent
            path.reverse()
            return (np.array(path), nodes_expanded)
        
        neighbors = get_neighbors(current_node, occ_map, goal_node, astar)
        for neighbor in neighbors:
            if neighbor not in cost_map or cost_map[neighbor] > neighbor.cost:
                neighbor.parent = current_node
                cost_map[neighbor] = neighbor.cost
                heappush(priority_queue, neighbor)

        nodes_expanded += 1

    # Return a tuple (path, nodes_expanded)
    return (None, 0)
