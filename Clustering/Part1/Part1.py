import sys
import os 
import copy
import pandas as pd
import numpy as np 
from anytree import Node, RenderTree
import matplotlib.pyplot as plt 
eps = np.finfo(float).eps
np.set_printoptions(suppress=True , formatter={'float': '{: 0.1f}'.format}, threshold=sys.maxsize )
pd.set_option('display.width', None)
standard_output = sys.stdout 

Clusters_Needed = 9 
Aux = pd.read_csv( "AAAI.csv" )

Features = Aux.keys().tolist()
Len = len(Aux)

Attr = "Topics"
Mapping, RevMapping, Root, Topic = {}, {}, {}, {}
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
    elem_Leaf = set()
    elem_Leaf.add(i)
    Root[i] = Node( name=str(i), parent=None, elements=elem_Leaf, left=None, right=None )    
    
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
        Closest[0][i] = np.nanargmax(Proximity[i])
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
        PrintDendo(Cluster_Roots[root]) 
        Count += 1
  
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
        Closest[0][i] = np.nanargmax(Proximity[i])
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
        Closest[0][I2] = -1 
        Closest[1][I2] = -1
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
        PrintDendo(Cluster_Roots[root]) 
        Count += 1        

    Clusters = [] 
    for root in Cluster_Roots:
        Clusters.append(Cluster_Roots[root].elements)
    return Clusters

MIN_ClusterTopics = copy.deepcopy(Topic)
MIN_ClusterRoots = copy.deepcopy(Root)
sys.stdout = open("SingleLinkage.txt", "w", encoding="utf-8")
MIN_Dendogram = SingleLinkage( MIN_ClusterTopics, MIN_ClusterRoots, Clusters_Needed )
sys.stdout = standard_output

MAX_ClusterTopics = copy.deepcopy(Topic)
MAX_ClusterRoots = copy.deepcopy(Root)
sys.stdout = open("CompleteLinkage.txt", "w", encoding="utf-8")
MAX_Dendogram = CompleteLinkage( MAX_ClusterTopics, MAX_ClusterRoots, Clusters_Needed )
sys.stdout = standard_output

    
 

