import numpy as np
import pandas as pd
from queue import Queue
import collections
from collections import deque

df = pd.read_csv("AIRPORT_DISTANCE.csv")
df1 = pd.DataFrame(columns = ['origin', 'dest', 'dist'])
df1['origin'] = df['Origin Airport Code']
df1['dest'] = df['Dest Airport Code']
df1['dist'] = df['Distance (Miles)']
#creating a new dataframe to store only the requried columns i.e., origin airport code, destinatino airport code and the distance between those two airports

cols = ['origin', 'dest']
cities = pd.factorize(df1[cols].values.ravel())[1]
cities_hm = {}
cities_hm_reverse = {}
tmp = 1

for city in cities:
  cities_hm[tmp] = city
  tmp+=1

tmp = 1
for city in cities:
  cities_hm_reverse[city] = tmp
  tmp+=1

df1[cols] = (pd.factorize(df1[cols].values.ravel())[0]+1).reshape(-1, len(cols))
df1 = df1.dropna(subset = ['dist'])

df1['origin'] = df1['origin'].apply(np.int64)
df1['dest'] = df1['dest'].apply(np.int64)
df1['dist'] = df1['dist'].apply(np.int64)

flights = df1.values.tolist()

n = len(cities)+1
m = len(flights)

from collections import defaultdict
class Graph:
  def __init__(self, no_of_nodes, directed=True):
    self.m_no_of_nodes = no_of_nodes
    self.m_nodes = range(self.m_no_of_nodes)
    self.m_directed = directed
    self.m_adj_list = {node: set() for node in self.m_nodes}

  def addEdge(self, u, v, w):
    self.m_adj_list[u].add((v,w))

  def bfs(self, start_node, target_node, k):
    print("start node is: ", start_node)
    print("target node is: ", target_node)
    visited = set()
    queue = Queue()
    #initiating the total cost of the shortest path as zero
    cost = 0

    #adding the start_node to the queue and visited list
    queue.put(start_node)
    visited.add(start_node)
    #printing the queue at this instant
    #queue.print()

    parent = dict()
    parent[start_node] = None

    found = False
    while not queue.empty():
      current_node = queue.get()
      if current_node == target_node:
        found = True
        break
    
      for(next_node, weight) in self.m_adj_list[current_node]:
        if next_node not in visited:
          queue.put(next_node)
          parent[next_node] = current_node
          visited.add(next_node)

     #path reconstruction
    while k >=0:
      if found:
        print("node found")
        path.append(target_node)
        print("Path till now is: ", path)
        while parent[target_node] is not None:
          print("parent of the above target node is: ", parent[target_node])
          path.append(parent[target_node])
          print("Path till now is: ", path)
          target_node = parent[target_node]
          print("now the updated target node is: ", target_node)
          print("Path till now is: ", path)
        path.reverse()
        print("reversed path is: ", path)
      k -= 1
      return path

g = Graph(m)
for source, destination, price in flights:
  g.addEdge(source, destination, price)


src = cities_hm_reverse['01A']
dest = cities_hm_reverse['MUC']
path = []
print("Initially, path is: ", path)
path = g.bfs(src, dest, 4)

current = path[0]
i=1
distance = 0
path_final = []
while i<len(path):
  path_final.append(cities_hm[current])
  previous = current
  current = path[i]
  
  for source, destination, price in flights:
    if source == previous and destination == current:
      distance = distance + price
  i = i+1
  if current == dest:
    path_final.append(cities_hm[current])
    break

print('Path: ',' -> '.join(path_final))
print("Distance is: ", distance)