
from typing import List, Dict, Iterable, Tuple, Set
from collections import defaultdict
import copy
from data_generation import simulate_tra_movement, rev_comp, print_orig, print_with_markers
from pulp import LpMinimize, LpProblem, LpVariable, PULP_CBC_CMD
from itertools import combinations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

score_matrix = {'A': {'A': 0, 'C': 2, 'G': 1, 'T': 3},
                'C': {'A': 4, 'C': 0, 'G': 2, 'T': 1},
                'G': {'A': 3, 'C': 2, 'G': 0, 'T': 5},
                'T': {'A': 1, 'C': 3, 'G': 4, 'T': 0}}

def find_Tra_in_population(population, tra_length_range = [10, 20], tsd_length_range = [3, 5], rev_comp_length_range = [4, 6]):
  """
  This function finds the transposon in the population of DNA sequences.
  Input:
      population: list of DNA sequences strings starting from DNA1.
      tra_length_range: list of two integers, the minimum and maximumx length of the transposon.
      tsd_length_range: list of two integers, the minimum and maximum length of the TSD.
      rev_comp_length: list of two integers, the minimum and maximum length of the reverse complement of the Transposon.

  Output:
      If the transposon is found, it returns a tuple of three elements:
      - the TSD sequence
      - the index of the DNA sequence where TSD is found, starting from 0
      - TSD position
      Otherwise, it returns None."""

  min_length = tra_length_range[0] + 2 * tsd_length_range[0]
  for k in range(len(population)):
    dna = population[k]
    for v in range(rev_comp_length_range[1], rev_comp_length_range[0] - 1, -1):
      for l in range(tra_length_range[1] + 1, max(2 * v, tra_length_range[0]) - 1, -1):
        for i in range(len(dna) - min_length):
          if rev_comp(dna[i : i + v]) == dna[(i + l - v) : (i + l)]:
            for t in range(tsd_length_range[1], tsd_length_range[0] -1, -1):
              if i + l + t - 1 < len(dna):
                if dna[i - t : i] == dna[(i + l) : (i + l + t)]:
                  print(f"Trasposon is '{dna[i : i + l]}' in DNA {k+1} at position {i-t} with {l} dp length.")
                  print(f"The TSD is '{dna[i - t : i]}' with {t} dp length.\n")
                  return (dna[i - t : i], k, i - t)
  return None

def find_TSDs_in_population(population, TSD):
    """
    This function finds all possible TSDs in the population of DNA sequences.
    Input:
        population: list of DNA sequences strings starting from 1 to n.
        TSD: the TSD found in the ending dna string.

    Output:
        list of tuples, each tuple contains:
        - the TSD sequence
        - the index of the DNA sequence where TSD is found
        - TSD position.
        Otherwise, show "No TSDs found" warning and return None."""

    TSDs = []
    tsd_length = len(TSD[0])
    for k in range(len(population)):
      if k == TSD[1]:
          continue
      dna = population[k]
      for i in range(len(dna) - 2*tsd_length):
        if dna[i : (i + tsd_length)] == dna[(i + tsd_length) : (i + 2*tsd_length)]:
          TSDs.append((dna[i : (i + tsd_length)], k, i))
        #   print(f"The possible TSD is '{dna[i : (i + tsd_length)]}' in DNA {k+1} at position {i}.")
    if TSDs == []:
      print("No TSDs found.")
      return None
    TSDs.insert(0, TSD)
    return TSDs

def ComputeScore(TSD1, TSD2, score_matrix):
    """
    This function computes the score of TSD1 mutated to TSD2.
    Input:
        TSD1, TSD2: list of tuples, each tuple contains:
        - the TSD sequence
        - the index of the DNA sequence where TSD is found
        - TSD position
        score_matrix: dictionary of dictionaries, the score matrix of different mutations from row value to column value.

    Output:
        It returns a float number as the score of TSD1 mutated to TSD2, which calculated by the given score matrix and normalized distance between two TSD pairs."""

    score = 0
    for i in range(len(TSD1[0])):
        score += score_matrix[TSD1[0][i]][TSD2[0][i]]
    if TSD1[2] == TSD2[2]:
        distance_norm = 0
    else:
        distance_norm = 2 * abs(TSD1[2] - TSD2[2]) / (TSD1[2] + TSD2[2])
    # print(f"The score of {TSD1} mutated to {TSD2} is {score + distance_norm}.")
    return score + distance_norm


def construct_graph(TSDs: List[Tuple[str, int, int]], undirected: bool) -> Tuple[Dict[int, int], Dict[Tuple[int, int], float]]:
    graph = defaultdict(list)
    weight = {}
    for i in range(len(TSDs)):
        for j in range(i+1, len(TSDs)):
            if TSDs[i][1] == TSDs[j][1]: # in the same part (DNA)
                continue
            dis = hamming_distance(TSDs[i][0], TSDs[j][0])
            if dis <= 1:
                u, v = min(i, j), max(i, j)
                graph[u].append(v)
                score = ComputeScore(TSDs[u], TSDs[v], score_matrix)
                weight[(u, v)] = score
                if undirected == False:
                    score = ComputeScore(TSDs[v], TSDs[u], score_matrix)
                    graph[v].append(u)
                    weight[(v, u)] = score
    return graph, weight

def hamming_distance(TSD1: str, TSD2: str) -> int:
    assert len(TSD1) == len(TSD2), 'Strings must have equal length when calculating the Hamming distance. '
    return sum([0 if TSD1[i]==TSD2[i] else 1 for i in range(len(TSD1))])

def traverse_graph(TSDs: List[Tuple[str, int, int]], n: int) -> List[List[int]]:
    graph, weight = construct_graph(TSDs, False)
    part = [TSD[1] for TSD in TSDs]
    path, paths = [], []
    visited = set()
    dfs(0, graph, n, part, visited, path, paths)
    results = []
    # print('All possible paths with score:')
    min_score = float('inf')
    for path in paths:
        p = path[::-1]
        results.append([path_score(p, weight), p])
        min_score = min(min_score, results[-1][0])
        # print(results[-1])
    return list(filter(lambda x: x[0]==min_score, results))

def dfs(v: int, graph: Dict[int, List[int]], n: int, part: List[int], \
        visited: set, path: List[int], paths: List[List[int]]):
    visited.add(part[v])
    path.append(v)
    if len(path) == n:
        paths.append(path[:])
    else:
        for u in graph[v]:
            if part[u] not in visited:
                dfs(u, graph, n, part, visited, path, paths)
    visited.remove(part[v])
    path.pop()

def path_score(path: List[int], weight: Dict[Tuple[int, int], float]) -> float:
    score = 0
    for i in range(len(path)-1):
        score += weight[(path[i], path[i+1])]
    return score

# Use ILP to solve the set TSP problem (NP hard)
def ILP(n: int, TSDs: List[Tuple[str, int, int]]) -> List[int]:
    graph, weight = construct_graph(TSDs, False)
    # Convert the Hamiltonian path problem to the Hamiltonian cycle problem (TSP)
    # Add a dummy node and virtual edges
    dummy, dummy_id = ('dummy', n, 0), len(TSDs)
    TSDs.append(dummy)
    graph[dummy_id] = []

    part = defaultdict(list)
    for i in range(len(TSDs)):
        # Add node to part
        part[TSDs[i][1]].append(i)
        # Add virtual edges to dummy node with weight 0
        if i not in graph[dummy_id]:
            if i != dummy:
                graph[dummy_id].append(i)
                weight[(dummy_id, i)] = 0
        if i not in graph:
            graph[i] = []
        if dummy_id not in graph[i]:
            graph[i].append(dummy_id)
            weight[(i, dummy_id)] = 0
    # Create the model
    model = LpProblem(name='Find_traversing_path', sense=LpMinimize)

    # Initialize the decision variables
    objectctive = 0
    edge2variable = {}
    for parent, children in graph.items():
        for child in children:
            edge2variable[(parent, child)] = LpVariable(name=f'{parent}_{child}', lowBound=0, upBound=1, cat="Integer")
            objectctive += edge2variable[(parent, child)] * weight[(parent, child)]
    
    # Add the obejctive function to the model
    model += objectctive

    # Add the constraints to the model
    # constraint 1: each part has 1 indegree and 1 outdegree
    for p, nodes in part.items():
        expression1, expression2 = 0, 0
        for i in nodes:
            for j in graph[i]:
                expression1 += edge2variable[(i, j)]
                expression2 += edge2variable[(j, i)]
        model += (expression1 == 1, f'constraint_{p}_outdegree')
        model += (expression2 == 1, f'constraint_{p}_indegree')
    # constraint 2: each part only has one node with degree > 0
    for p, nodes in part.items():
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                expression = 0
                for k in graph[i]:
                    expression += edge2variable[(i, k)]
                for k in graph[j]:
                    expression += edge2variable[(k, j)]
                model += (expression <= 1, f'constraint_{i}_{j}')
    # constraint 3: for each edge, only one direction can be selected
    visited = []
    for parent,children in graph.items():
        for child in children:
            if (parent, child) in visited:
                continue
            visited.append((parent, child))
            visited.append((child, parent))
            expression = edge2variable[(parent, child)] + edge2variable[(child, parent)]
            model += (expression <= 1, f'constraint_edge_{(parent, child)}')
    # constraint 4: there is an edge between node v and the dummy node
    expression = edge2variable[(0, dummy_id)]
    model += (expression == 1, f'start_and_end')
    
    # constraint 5: connectivity between any bipartite subsets
    all_parts = list(part.keys())
    sets = [com for sub in range(len(all_parts)) for com in combinations(all_parts, sub + 1)]
    for k in range(len(sets)):
        s = sets[k]
        node_set1 = [node for i in s for node in part[i]]
        node_set2 = [node for i in (set(all_parts)-set(s)) for node in part[i]]
        expression = 0
        for i in node_set1:
            for j in node_set2:
                if (i, j) in edge2variable:
                    expression += edge2variable[(i, j)] + edge2variable[(j, i)]
        if expression != 0:
            model += (expression >= 2, f'connectivity_{k}')

    # solve the ILP
    status = model.solve(PULP_CBC_CMD(msg=0))
    edge = {}
    for var in model.variables():
        if var.value() > 0:
            # print(var)
            v, u = var.name.split('_')
            edge[int(v)] = int(u)
    path = [0]
    node = edge[0]
    while node != 0:
        path.append(node)
        node = edge[node]
    # print(path)
    if path[-1] == dummy_id:
        path = path[:-1]
    else:
        path = [0] + path[2:][::-1]
    # print(part, dummy_id)
    path = path[::-1]
    return [path_score(path, weight), path]

def ILP1(n: int, TSDs: List[Tuple[str, int, int]]) -> List[int]:
    graph, weight = construct_graph(TSDs, True)
    # Convert the Hamiltonian path problem to the Hamiltonian cycle problem (TSP)
    # Add a dummy node and virtual edges
    dummy, dummy_id = ('dummy', n, 0), len(TSDs)
    TSDs.append(dummy)

    part = defaultdict(list)
    for i in range(len(TSDs)):
        # Add node to part
        part[TSDs[i][1]].append(i)
        # Add virtual edges to dummy node with weight 0
        if i == dummy_id:
            continue
        if i not in graph:
            graph[i] = []
        if dummy_id not in graph[i]:
            graph[i].append(dummy_id)
            weight[(i, dummy_id)] = 0
    # print(graph, weight, part)
    # Create the model
    model = LpProblem(name='Find_traversing_path', sense=LpMinimize)

    # Initialize the decision variables
    objectctive = 0
    edge2variable = {}
    for parent, children in graph.items():
        for child in children:
            edge2variable[(parent, child)] = LpVariable(name=f'{parent}_{child}', lowBound=0, upBound=1, cat="Integer")
            objectctive += edge2variable[(parent, child)] * weight[(parent, child)]
    
    # Add the obejctive function to the model
    model += objectctive

    # Add the constraints to the model
    # constraint 1: each part has 1 indegree and 1 outdegree
    for p, nodes in part.items():
        expression = 0
        for (i, j) in weight.keys():
            if i in nodes or j in nodes:
                expression += edge2variable[(i, j)]
        model += (expression == 2, f'constraint_{p}_outdegree')
    # constraint 2: each part only has one node with degree > 0
    for p, nodes in part.items():
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                expression = 0
                for k in graph[i]:
                    expression += edge2variable[(i, k)]
                for k in graph[j]:
                    expression += edge2variable[(j, k)]
                model += (expression <= 1, f'constraint_{i}_{j}')
    # constraint 4: there is an edge between node v and the dummy node
    expression = edge2variable[(0, dummy_id)]
    model += (expression == 1, f'start_and_end')
    
    # connectivity
    all_parts = list(part.keys())
    sets = [com for sub in range(len(all_parts)) for com in combinations(all_parts, sub + 1)]
    for k in range(len(sets)):
        s = sets[k]
        print(set(s), set(all_parts)-set(s))
        node_set1 = [node for i in s for node in part[i]]
        node_set2 = [node for i in (set(all_parts)-set(s)) for node in part[i]]
        expression = 0
        for i in node_set1:
            for j in node_set2:
                if (i, j) in edge2variable:
                    expression += edge2variable[(i, j)]
        if expression != 0:
            model += (expression >= 2, f'connectivity_{k}')

    # solve the ILP
    status = model.solve()
    edge = {}
    for var in model.variables():
        if var.value() > 0:
            print(var)
            v, u = var.name.split('_')
            edge[int(v)] = int(u)
    path = [0]
    node = edge[0]
    while node != 0:
        path.append(node)
        node = edge[node]
    if path[-1] == dummy_id:
        path = path[:-1]
    else:
        path = [0] + path[2:][::-1]
    return [path_score(path, weight), path]

def benchmark():
    import csv
    sample_id = 1
    for n in [3,4,5,6,7,8,9,10,12,15]:
        for _ in range(10):
            print(f'simulating sample {sample_id}')
            orig_tsd, tra, original_population, population, insertion_path, insertion_details = simulate_tra_movement(n, dna_length_range = [50, 200], tra_length_range = [10, 20], tsd_length_range = [3, 8], p = 0.5)
            with open(f'data/population_{sample_id}.csv', 'w') as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow([n, len(tra), len(orig_tsd)])
                for t in insertion_path:
                    csvwriter.writerow(list(t))
                csvwriter.writerow([])
                for p in population:
                    csvwriter.writerow([p])
                f.close()
            TSD = insertion_path[-1]
            TSDs = find_TSDs_in_population(population, TSD)
            with open(f'data/sample_{sample_id}.csv', 'w') as f:
                csvwriter = csv.writer(f)
                for t in TSDs:
                    csvwriter.writerow(list(t))
                f.close()
            sample_id += 1

def testing():
    import time
    import csv
    # csvwriter = csv.writer(open('results.csv', 'a'))
    # csvwriter.writerow(['sample_id', 'num_of_DNA', 'num_of_TSD', 'num_of_edges', 'score', 'time', 'method'])
    for sample_id in range(15, 16):
        print(f'solving sample {sample_id}')
        TSDs = []
        for line in open(f'data/sample_{sample_id}.csv', 'r').readlines():
            # print(line)
            tsd, dna_id, p = line.rstrip().split(',')
            TSDs.append((tsd, int(dna_id), int(p)))
        n = TSDs[0][1]+1

        # simulation 
        simulated_path = []
        for line in open(f'data/population_{sample_id}.csv', 'r').readlines()[1:]:
            if line == '\n':
                break
            tsd, dna_id, p = line.rstrip().split(',')
            simulated_path.append((tsd, int(dna_id), int(p)))
        score = 0
        for i in range(len(simulated_path)-1):
            score += ComputeScore(simulated_path[i], simulated_path[i+1], score_matrix)
        graph, weight = construct_graph(TSDs, True)
        print(graph, weight, TSDs)
        num_edges = len(weight)
        # csvwriter.writerow([sample_id, n, len(TSDs), num_edges, score, 0, 'simulation'])
        # DFS
        start_time = time.time()
        paths = traverse_graph(TSDs, n)
        end_time = time.time()
        # csvwriter.writerow([sample_id, n, len(TSDs), num_edges, paths[0][0], end_time - start_time, 'DFS'])
        # ILP
        start_time = time.time()
        path = ILP(n, TSDs[:])
        end_time = time.time()
        # csvwriter.writerow([sample_id, n, len(TSDs), num_edges, path[0], end_time - start_time, 'ILP'])

def score_plot():
    d1 = pd.read_csv('results.csv', header=0)
    sns.set_theme()
    sns.lineplot(data=d1, x="sample_id", y="score", hue="method", style="method", markers=["s", "o", "."])
    plt.savefig('score.png')

def time_plot():
    d2 = pd.read_csv('results.csv', header=0)
    d2 = d2.loc[d2['method'] != 'simulation']
    d2 = d2.loc[d2['num_of_edges'] < 100]
    sns.lineplot(data=d2, x="num_of_edges", y="time", hue="method", style="method", markers=True)
    plt.savefig('time_edge.png')

if __name__ == "__main__":
    # benchmark()
    # testing()
    # score_plot()
    # time_plot()
    TSDs = [('ACT', 0, 20), ('CCT', 1, 30), ('ACC', 1, 80), ('AGT', 1, 180), ('CTT', 2, 70), ('ACT', 2, 80)]
    score_matrix = {'A': {'A': 0, 'C': 4, 'G': 3, 'T': 1},
                'C': {'A': 4, 'C': 0, 'G': 2, 'T': 3},
                'G': {'A': 3, 'C': 2, 'G': 0, 'T': 5},
                'T': {'A': 1, 'C': 3, 'G': 5, 'T': 0}}
    graph, weight = construct_graph(TSDs, True)
    print(graph, weight)
    n = 3
    paths = traverse_graph(TSDs, n)
    print('DFS', paths)
    path = ILP(n, TSDs[:])
    print('ILP', path)
    transposition = []
    for i in path[1]:
        transposition.append(TSDs[i])
    print('transposition', transposition)
    
    