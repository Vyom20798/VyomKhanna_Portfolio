#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 02:06:31 2024

@author: houziwu
"""

def dijkstra(graph, start):
    '''
    Given the graph containing distances between different nodes and the starting node,
    it returns a shortest time-taking route to travel all the nodes on the graph.
    
    Keyword arguments:
    graph (dictionary) -- A dictionary representing the graph where keys are nodes and
                          values are dictionaries of neighboring nodes with their corresponding distances.
    start (string) -- The name of the starting node.
    
    Returns:
    result (dictionary): A dictionary containing the shortest time-taking route information.
          Each node is a key with a sub-dictionary containing 'distance' (total distance to reach the node)
          and 'path' (the shortest path to reach the node from the starting node).
    '''
    # Initialize distances with infinity for all nodes except the start node
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0

    # Use a list to keep track of nodes with the smallest distances
    priority_queue = [(0, start)]

    # Track the predecessors for each node
    predecessors = {node: None for node in graph}

    while priority_queue:
        # Find the node with the smallest distance in the priority queue
        current_distance, current_node = min(priority_queue)

        # Remove the current node from the priority queue
        priority_queue.remove((current_distance, current_node))

        # Check if the current distance is already greater than the known distance
        if current_distance > distances[current_node]:
            continue

        # Iterate through neighbors of the current node
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            # Update the distance and predecessor if a shorter path is found
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                priority_queue.append((distance, neighbor))

    # Build and return the result including distances and paths
    result = {node: {'distance': distances[node], 'path': get_path(start, node, predecessors)} for node in graph}
    return result

def get_path(start, end, predecessors):
    path = []
    current_node = end
    while current_node is not None:
        path.insert(0, current_node)
        current_node = predecessors[current_node]
    return path


# Testing of algorithm on a smaller problem as a coding sanity check
test_graph = {
    'A': {'B': 10, 'C': 3},
    'B': {'C': 1, 'D': 2},
    'C': {'B': 4, 'D': 8, 'E': 2},
    'D': {'E': 7},
    'E': {'D': 9}
}

starting_node = 'A'
destination_node = 'D'
result = dijkstra(test_graph, starting_node)

distance = result[destination_node]['distance']
path = result[destination_node]['path']
print(f"Starting station : {starting_node}")
print(f"Destination station : {destination_node}")
print(f"Time : {distance} minutes")
print(f"Route : {path}")


# Import the graph of London underground:
London_graph = {
    'Paddington': {'Notting Hill Gate': 4, 'Baker Street': 6},
    'Notting Hill Gate': {'Paddington': 4, 'Bond Street': 7, 'South Kensington': 7},
    'South Kensington': {'Notting Hill Gate': 7, 'Green Park': 7, 'Victoria': 4},
    'Baker Street': {'Paddington': 6, 'Bond Street': 2, 'Oxford Circus': 4, 'Kings Cross': 7},
    'Bond Street': {'Notting Hill Gate': 7, 'Baker Street': 2, 'Green Park': 2, 'Oxford Circus': 1},
    'Green Park': {'South Kensington': 7, 'Bond Street': 2, 'Victoria': 2, 'Oxford Circus': 2, 'Piccadilly Circus': 1, 'Westminster': 3},
    'Victoria': {'South Kensington': 4, 'Green Park': 2, 'Westminster': 4},
    'Oxford Circus': {'Baker Street': 4, 'Bond Street': 1, 'Green Park': 2, 'Piccadilly Circus': 2, 'Warren Street': 2, 'Tottenham Court Road': 2},
    'Piccadilly Circus': {'Green Park': 1, 'Oxford Circus': 2, 'Leicester Square': 2, 'Charing Cross': 2},
    'Westminster': {'Green Park': 3, 'Victoria': 4, 'Embankment': 2, 'Waterloo': 2},
    'Warren Street': {'Oxford Circus': 2, 'Tottenham Court Road': 3, 'Kings Cross': 3},
    'Tottenham Court Road': {'Oxford Circus': 2, 'Warren Street': 3, 'Leicester Square': 1, 'Holborn': 2},
    'Leicester Square': {'Piccadilly Circus': 2, 'Tottenham Court Road': 1, 'Charing Cross': 2, 'Holborn': 2},
    'Charing Cross': {'Piccadilly Circus': 2, 'Leicester Square': 2, 'Embankment': 1},
    'Embankment': {'Westminster': 2, 'Charing Cross': 1, 'Waterloo': 2, 'Blackfriars': 4},
    'Waterloo': {'Westminster': 2, 'Embankment': 2, 'Elephant and Castle': 4},
    'Holborn': {'Tottenham Court Road': 2, 'Leicester Square': 2, 'Kings Cross': 4, 'Bank': 5},
    'Elephant and Castle': {'Waterloo': 4, 'London Bridge': 3},
    'Kings Cross': {'Baker Street': 7, 'Warren Street': 3, 'Holborn': 4, 'Old Street': 6, 'Moorgate': 6},
    'Blackfriars': {'Embankment': 4, 'Bank': 4},
    'Old Street': {'Kings Cross': 6, 'Moorgate': 1},
    'Moorgate': {'Kings Cross': 6, 'Old Street': 1, 'Bank': 3, 'Liverpool Street': 2},
    'Bank': {'Holborn': 5, 'Blackfriars': 4, 'Moorgate': 3, 'London Bridge': 2, 'Liverpool Street': 2, 'Tower Hill': 2},
    'London Bridge': {'Waterloo': 3, 'Elephant and Castle': 3, 'Bank': 2},
    'Liverpool Street': {'Moorgate': 2, 'Bank': 2, 'Tower Hill': 6, 'Aldgate East': 4},
    'Tower Hill': {'Bank': 2, 'Liverpool Street': 6, 'Aldgate East': 2},
    'Aldgate East': {'Liverpool Street': 4, 'Tower Hill': 2}
}

starting_station = 'Paddington'
destination_station = 'Charing Cross'
result = dijkstra(London_graph, starting_station)

distance = result[destination_station]['distance']
path = result[destination_station]['path']
print(f"Starting station : {starting_station}")
print(f"Destination station : {destination_station}")
print(f"Time : {distance} minutes")
print(f"Route : {path}")


