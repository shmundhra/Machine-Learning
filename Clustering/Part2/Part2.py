import sys
import os 
import copy
import pandas as pd
import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt
eps = np.finfo(float).eps
np.set_printoptions(suppress=True , formatter={'float': '{: 0.1f}'.format}, threshold=sys.maxsize )
pd.set_option('display.width', None)
standard_output = sys.stdout 

Clusters_Needed = 9 
Threshold = [0.10, 0.12, 0.15, 0.17, 0.20, 0.25, 0.30] 
Aux = pd.read_csv( "AAAI.csv" )

Features = Aux.keys().tolist()
Len = len(Aux)

Attr = "Topics"
Mapping, RevMapping, Topic = {}, {}, {}
G = nx.Graph() 
Count = 0 
for i in range(Len):
    Topic[i] = set()
    topics = Aux[Attr][i].split("\n")
    for temp in topics:
        if not temp in Mapping:
            Mapping[temp] = Count
            RevMapping[Count] = temp
            Count += 1
        Topic[i].add(Mapping[temp])
    element = set()
    element.add(i)
    G.add_node( i , topics=Topic[i], elements=element )

def Jaccard_Coeff( A , B ):
    Intersection = A.intersection(B)
    Union = A.union(B)
    Card_Intersection = float(len(Intersection))
    Card_Union = float(len(Union))
    return Card_Intersection/(Card_Union+eps)

def GraphClustering( G, NodeTopics, threshold ):

    Proximity = np.zeros( shape=(Len, Len) )
    for i in range(Len):
        for j in range(Len):
            if (i == j) :
                continue
            Proximity[i][j] = Jaccard_Coeff( NodeTopics[i], NodeTopics[j] )
            if ( Proximity[i][j] > threshold ):
                G.add_edge( i , j )

    pd.DataFrame(Proximity).to_csv(r"Proximity.csv")
                
    print("\nGRAPH CLUSTER with THRESHOLD for Edge (Similarity) between two Nodes (Papers) = %.2f\n" % threshold )
    print("INITIAL NUMBER of CONNECTED COMPONENTS = %d" % nx.number_connected_components(G) )            
    print("FINAL CLUSTERS : \n")

    while nx.number_connected_components(G) < Clusters_Needed :
        BetweenNess = nx.edge_betweenness_centrality(G)
        Max = np.NINF
        MaxEdge = -1   
        for edge in BetweenNess:
            if ( BetweenNess[edge] > Max ):
                Max = BetweenNess[edge]
                MaxEdge = edge

        G.remove_edge(MaxEdge[0], MaxEdge[1]) 
    
    Count = 1 
    for cluster in nx.connected_components(G):
        print( ("CLUSTER %d WITH %d OBJECTS =")%(Count,len(cluster)) , end = ' ' )
        print(cluster, end = "\n\n" )
        Count += 1
    
    Clusters = []
    for cluster in nx.connected_components(G):
        Clusters.append(cluster)
    return Clusters 

for threshold in Threshold:
    NodeTopics = copy.deepcopy(Topic)
    Graph = copy.deepcopy(G) 
    sys.stdout = open( ("GirvanNewmann_%d.txt")%(100*threshold) , "w" , encoding="utf-8")
    GraphCluster1 = GraphClustering( Graph, NodeTopics, threshold )
    sys.stdout = standard_output