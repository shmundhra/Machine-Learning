import sys
import os 
import copy
import pandas as pd
import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt
from anytree import Node, RenderTree
from numpy import log2 
np.set_printoptions(suppress=True , formatter={'float': '{: 0.6f}'.format}, threshold=sys.maxsize )
pd.set_option('display.width', None)
eps = np.finfo(float).eps
err = 1e-8
standard_output = open("Results.txt", "w", encoding="utf-8")
sys.stdout = standard_output 

Clusters_Needed = 9 
Threshold = [0.1, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.2, 0.25, 0.30 ] 
Aux = pd.read_csv( "AAAI.csv" )

Features = Aux.keys().tolist()
Len = len(Aux)

Attr = "Topics"
Mapping, RevMapping, Root, Topic = {}, {}, {}, {} 
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
    Root[i] = Node( name=str(i), parent=None, elements=element, left=None, right=None )    
    G.add_node( i , topics=Topic[i], elements=element )

Class = "High-Level Keyword(s)"
Unique_Labels, ClassLabels = np.unique(Aux[Class].values, return_inverse=True )

def Jaccard_Coeff( A , B ):
    Intersection = A.intersection(B)
    Union = A.union(B)
    Card_Intersection = float(len(Intersection))
    Card_Union = float(len(Union))
    return Card_Intersection/(Card_Union+eps)

def PrintDendo(Root):
    for pre, fill, node in RenderTree(Root):
        Dendogram = pre + node.name.strip() 
        print(Dendogram)
    print("_"*100)
        
def Entropy ( Probability ):
    H = 0
    for p in Probability:
        H -= p*log2(p+eps)
    if H < err : 
        H = 0 
    return H

def Calc_Entropy ( Class_Val ) :
    Total = len(Class_Val) 
    Labels, Count = np.unique(Class_Val, return_counts=True )
    Probability = Count/Total 
    return Entropy(Probability) 

def CalcNMI( Clusters ):
    H_Class = Calc_Entropy(ClassLabels) 
    
    Total = 0 
    Cluster_Size = []
    for cluster in Clusters:
        Cluster_Size.append(len(cluster))
        Total += Cluster_Size[-1] 
    Probability = np.array(Cluster_Size)/Total
    H_Cluster = Entropy(Probability)

    H_ClassCluster = 0 
    Cluster_Info = [] 
    for i in range(len(Clusters)):
        Cluster_ClassLabels = [] 
        for paper in Clusters[i]:
            Cluster_ClassLabels.append( ClassLabels[paper] ) 
        Cluster_Info.append( Calc_Entropy(Cluster_ClassLabels) ) 

        H_ClassCluster += Probability[i] * Cluster_Info[i]

    MutualInfo = H_Class - H_ClassCluster 
    NMI = (2 * MutualInfo)/(H_Class+H_Cluster) 
    return NMI 

def SingleLinkage( Cluster_Topics, Cluster_Roots,  Reqd_ClusterCount ):
    
    Proximity = np.full( [Len, Len] , np.nan )
    for i in range(Len):
        for j in range(Len):
            if (i == j) :
                continue
            Proximity[i][j] = Jaccard_Coeff( Topic[i], Topic[j] )
    
    pd.DataFrame(Proximity).to_csv(r"Proximity.csv")

    Closest = [ [-1]*Len, [-1]*Len ] 
    for i in range(Len):
        Closest[0][i] = int(np.nanargmax(Proximity[i]))
        Closest[1][i] = Proximity[i][ Closest[0][i] ]

    Curr_ClusterCount = Len 
    while (Curr_ClusterCount > Reqd_ClusterCount) :
        i1 = np.nanargmax(Closest[1])
        i2 = Closest[0][i1] 
        I1 = int(min(i1, i2))
        I2 = int(max(i1, i2))
        I1,I2 = I2,I1 

        #I2 will be Merged Into I1 now 
        Cluster_Topics[I1] = Cluster_Topics[I1].union(Cluster_Topics[I2])
        Cluster_Topics[I2].clear()
        Curr_ClusterCount -= 1 

        #Create the Node for the New Cluster and make necessary linkages
        elem_Union = (Cluster_Roots[I1].elements).union(Cluster_Roots[I2].elements) 
        root = Node( "|", parent=None, elements=elem_Union, left=Cluster_Roots[I1], right=Cluster_Roots[I2] )
        Cluster_Roots[I1].parent = Cluster_Roots[I2].parent = root 
        Cluster_Roots[I1] = root 
        Cluster_Roots.pop(I2, None)

        #Update Proximity Matrix 
        for j in range(Len):
            Proximity[I1][j] = max( Proximity[I1][j] , Proximity[I2][j] )
        for i in range(Len):
            Proximity[i][I1] = Proximity[I1][i] 
        Proximity[I1][I1] = np.nan
        for j in range(Len):
            Proximity[I2][j] = Proximity[j][I2] = np.nan

        #Update Closest Matrix 
        Closest[0][I2] = np.nan 
        Closest[1][I2] = np.nan
        for j in range(Len):
            if ( Closest[0][j] == I2 ):
                Closest[0][j] = I1 
        try:
            Closest[0][I1] = np.nanargmax(Proximity[I1]) 
            Closest[1][I1] = Proximity[I1][ Closest[0][I1] ]
        except: 
            print("Row %d of Proximity Matrix is all Nan : Clustering Complete" % I1 ) 
        
  
    Count = 1
    print("\n%d Clusters formed using Single Linkage on the AAAI Submitted Papers Dataset "%Reqd_ClusterCount, end='')
    print("using Jaccard Coefficient on “Topics” as the Parameter for measuring the Similarity of Two Papers") 
    for root in Cluster_Roots:
        print( (("\nCLUSTER %d INDEXED BY %d HAS %d OBJECTS =") % (Count , root, len(Cluster_Roots[root].elements) )), end = ' ' )
        print(Cluster_Roots[root].elements)        
        Count += 1
    print("_"*100)

    sys.stdout = open("SingleLinkage.txt", "w", encoding="utf-8")
    for root in Cluster_Roots:
        PrintDendo(Cluster_Roots[root]) 
    sys.stdout = standard_output

    Clusters = [] 
    for root in Cluster_Roots:
        Clusters.append(Cluster_Roots[root].elements)
    return Clusters 


def CompleteLinkage( Cluster_Topics, Cluster_Roots,  Reqd_ClusterCount ):
    
    Proximity = np.full( [Len, Len] , np.nan )
    for i in range(Len):
        for j in range(Len):
            if (i == j) :
                continue
            Proximity[i][j] = Jaccard_Coeff( Topic[i], Topic[j] )
    
    Closest = [ [-1]*Len, [-1]*Len ] 
    for i in range(Len):
        Closest[0][i] = int(np.nanargmax(Proximity[i]))
        Closest[1][i] = Proximity[i][ Closest[0][i] ]

    Curr_ClusterCount = Len 
    while (Curr_ClusterCount > Reqd_ClusterCount) :
        i1 = np.nanargmax(Closest[1])
        i2 = Closest[0][i1] 
        I1 = int(min(i1, i2))
        I2 = int(max(i1, i2))
        I1,I2 = I2,I1 

        Cluster_Topics[I1] = Cluster_Topics[I1].union(Cluster_Topics[I2])
        Cluster_Topics[I2].clear()
        Curr_ClusterCount -= 1 

        #Create the Node for the New Cluster and make necessary linkages
        elem_Union = (Cluster_Roots[I1].elements).union(Cluster_Roots[I2].elements) 
        root = Node( "|", parent=None, elements=elem_Union, left=Cluster_Roots[I1], right=Cluster_Roots[I2] )
        Cluster_Roots[I1].parent = Cluster_Roots[I2].parent = root 
        Cluster_Roots[I1] = root 
        Cluster_Roots.pop(I2, None)

        #Update Proximity Matrix 
        for j in range(Len):
            Proximity[I1][j] = min( Proximity[I1][j] , Proximity[I2][j] )
        for i in range(Len):
            Proximity[i][I1] = Proximity[I1][i] 
        Proximity[I1][I1] = np.nan
        for j in range(Len):
            Proximity[I2][j] = Proximity[j][I2] = np.nan

        #Update Closest Matrix 
        Closest[0][I2] = np.nan 
        Closest[1][I2] = np.nan
        for j in range(Len):
            if ( Closest[0][j] == I2 ):
                Closest[0][j] = I1 
        try:
            Closest[0][I1] = np.nanargmax(Proximity[I1]) 
            Closest[1][I1] = Proximity[I1][ Closest[0][I1] ]
        except:
            print("Row %d of Proximity Matrix is all Nan : Clustering Complete" % I1 ) 
        

    Count = 1 
    print("\n%d Clusters formed using Complete Linkage on the AAAI Submitted Papers Dataset "%Reqd_ClusterCount, end='')
    print("using Jaccard Coefficient on “Topics” as the Parameter for measuring the Similarity of Two Papers")
    for root in Cluster_Roots:
        print( (("\nCLUSTER %d INDEXED BY %d HAS %d OBJECTS =") % (Count , root, len(Cluster_Roots[root].elements) )), end = ' ' )
        print(Cluster_Roots[root].elements)        
        Count += 1        
    print("_"*100)   
    
    sys.stdout = open("CompleteLinkage.txt", "w", encoding="utf-8")
    for root in Cluster_Roots:
        PrintDendo(Cluster_Roots[root]) 
    sys.stdout = standard_output 

    Clusters = [] 
    for root in Cluster_Roots:
        Clusters.append(Cluster_Roots[root].elements)
    return Clusters  


def GraphClustering( Graph, NodeTopics, threshold ):

    Proximity = np.zeros( shape=(Len, Len) )
    for i in range(Len):
        for j in range(Len):
            if (i == j) :
                continue
            Proximity[i][j] = Jaccard_Coeff( NodeTopics[i], NodeTopics[j] )
            if ( Proximity[i][j] > threshold ):
                Graph.add_edge( i , j )

    print("\nGRAPH CLUSTER with THRESHOLD for Edge (Similarity) between two Nodes (Papers) = %.2f\n" % threshold )
    print("INITIAL NUMBER of CONNECTED COMPONENTS = %d" % nx.number_connected_components(G) )            
    print("FINAL CLUSTERS : \n")

    while nx.number_connected_components(Graph) < Clusters_Needed :
        BetweenNess = nx.edge_betweenness_centrality(Graph)
        Max = np.NINF
        MaxEdge = -1   
        for edge in BetweenNess:
            if ( BetweenNess[edge] > Max ):
                Max = BetweenNess[edge]
                MaxEdge = edge

        Graph.remove_edge(MaxEdge[0], MaxEdge[1]) 
    
    Count = 1 
    for cluster in nx.connected_components(Graph):
        print( ("CLUSTER %d WITH %d OBJECTS =")%(Count,len(cluster)) , end=' ' )
        print(cluster, end="\n\n")
        Count += 1
    print("_"*100)       

    Clusters = []
    for cluster in nx.connected_components(Graph):
        Clusters.append(cluster)
    return Clusters 

MIN_ClusterTopics = copy.deepcopy(Topic)
MIN_ClusterRoots = copy.deepcopy(Root)
MIN_Cluster = SingleLinkage( MIN_ClusterTopics, MIN_ClusterRoots, Clusters_Needed )

MAX_ClusterTopics = copy.deepcopy(Topic)
MAX_ClusterRoots = copy.deepcopy(Root)
MAX_Cluster = CompleteLinkage( MAX_ClusterTopics, MAX_ClusterRoots, Clusters_Needed )

GraphCluster = []
for threshold in Threshold:
    Graph = copy.deepcopy(G) 
    NodeTopics = copy.deepcopy(Topic)
    GraphCluster.append( GraphClustering(Graph, NodeTopics, threshold) )

ClusterMethod = []
NMI = [] 

ClusterMethod.append("SINGLE LINKAGE")
NMI.append( CalcNMI(MIN_Cluster) )

ClusterMethod.append("COMPLETE LINKAGE")
NMI.append( CalcNMI(MAX_Cluster) )

temp = "GRAPH CLUSTER ( %0.3f )"
for i in range(len(GraphCluster)):
    ClusterMethod.append( (temp%Threshold[i]) )
    NMI.append( CalcNMI( GraphCluster[i] ) )

frame = { "NMI Values" : NMI } 
DF = pd.DataFrame(data = frame)
DF.index = ClusterMethod
DF.to_csv(r"NMI_Values.csv")
















