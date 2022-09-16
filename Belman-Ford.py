import pandas as pd
import numpy as np

df = pd.read_csv("airport_distances1.csv")

df1 = pd.DataFrame(columns=['origin', 'dest', 'dist'])
df1['origin'] = df['Origin Airport Code']
df1['dest'] = df['Dest Airport Code']
df1['dist'] = df['Distance (Miles)']

cols = ['origin', 'dest']
# changing cities to categories and saving them to hashamps to retreive later
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


def bellamnFord(n, flights, src, dst, k):
    # initialising the dp array and setting src node cost to 0
    # print(src)
    dp = [float("inf")] * n
    dp[src] = 0

    for i in range(k+1):
        tmp = list(dp)
        # checking reachable nodes in every loop and updating the costs
        for source, destination, price in flights:

            if dp[source] == float("inf"):
                continue
            if dp[source]+price < tmp[destination]:
                tmp[destination] = dp[source]+price
                path[destination] = source
        dp = tmp

    if dp[dst] == float("inf"):
        return -1
    return dp[dst]


src = cities_hm_reverse['01A']
dest = cities_hm_reverse['KPY']
k = 2
print('least distance: ', bellamnFord(n, flights, src, dest, k))


current = dest
path_final = []
while current != None:
    path_final.append(cities_hm[current])
    current = path[current]
    if current == src:
        path_final.append(cities_hm[current])
        break

print('Path: ', ' -> '.join(reversed(path_final)))