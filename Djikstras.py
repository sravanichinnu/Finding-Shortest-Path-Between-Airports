import heapq
import pandas as pd
import numpy as np

df = pd.read_csv("airport_distances1.csv")

df1 = pd.DataFrame(columns=['origin', 'dest', 'dist'])
df1['origin'] = df['Origin Airport Code']
df1['dest'] = df['Dest Airport Code']
df1['dist'] = df['Distance (Miles)']

cols = ['origin', 'dest']
cities = pd.factorize(df1[cols].values.ravel())[1]
cities_hm = {}
cities_hm_reverse = {}
tmp = 1
for city in cities:
    cities_hm[tmp] = city
    tmp += 1
    
tmp = 1
for city in cities:
    cities_hm_reverse[city] = tmp
    tmp += 1

df1[cols] = (pd.factorize(df1[cols].values.ravel())
             [0]+1).reshape(-1, len(cols))


df1 = df1.dropna(subset=['dist'])

df1['origin'] = df1['origin'].apply(np.int64)
df1['dest'] = df1['dest'].apply(np.int64)
df1['dist'] = df1['dist'].apply(np.int64)

flights = df1.values.tolist()

#n = len(pd.unique(df['Origin Airport Code']))
n = len(cities)+1
path = {}


def djikstrasPath(n, flights, src, dst, K):
    # adjacency matrix
    adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for s, d, w in flights:
        adj_matrix[s][d] = w

    # Shortest distances array
    distances = [float("inf") for _ in range(n)]
    current_stops = [float("inf") for _ in range(n)]
    distances[src], current_stops[src] = 0, 0

    # Data is (cost, stops, node)
    minHeap = [(0, 0, src)]

    while minHeap:

        cost, stops, node = heapq.heappop(minHeap)

        # If dest is reached return the cost and the path to get here
        if node == dst:
            return cost, path

        # If we are out of stops, then continue
        if stops == K + 1:
            continue

        # check and drop all neighboring edges if possible
        for nei in range(n):
            if adj_matrix[node][nei] > 0:
                node_cost = cost
                neighbors_cost = distances[nei]
                adj_matrix_dst = adj_matrix[node][nei]

                # update if dist less that what we have
                if node_cost + adj_matrix_dst < neighbors_cost:
                    distances[nei] = node_cost + adj_matrix_dst
                    path[node] = nei
                    heapq.heappush(minHeap, (node_cost + adj_matrix_dst, stops + 1, nei))
                    current_stops[nei] = stops
                elif stops < current_stops[nei]:
                    #  add to the heap to check for better distances
                    heapq.heappush(minHeap, (node_cost + adj_matrix_dst, stops + 1, nei))
                    
                    
    if distances[dst] == float("inf"):
        return -1, -1
    else:
        distances[dst], path


src = cities_hm_reverse['01A']
dest = cities_hm_reverse['KPY']
k = 2

# print(djikstrasPath(n, flights, src, dest, k))
cost, path = djikstrasPath(n, flights, src, dest, k)

if cost == -1:
    print("Path Not poosible in given stops")
else:
    current = src
    path_final = []
    while current != None:
        path_final.append(cities_hm[current])
        current = path[current]
        if current == dest:
            path_final.append(cities_hm[current])
            break
    print("least distance: ", cost)
    print('Path: ', ' -> '.join((path_final)))
