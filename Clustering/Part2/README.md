# HIERARCHICAL DIVISIVE CLUSTERING 

In this folder, we aim to create a Clustering of the given Dataset of **Submitted Papers in the AAAI**, in a Top-Down Fashion using a Graph-Based Clustering Method. The Metric of Similarity between two Papers is the **Jaccard Coefficient** applied on the ***set of Topics*** which each Paper caters to. 
We have used the *Girvan Newmann Algorithm* to generate the Clustering. 


## DATA STRUCTURES and ALGORITHM 

We use a **Proximity Matrix**, which stores the Proximity 
between each pairs of *Paper Clusters*.  
We then add edges between each two Papers(Nodes) whose Proximity is greater than the threshold chosen.  	
After the Graph is formed in this way, we iteratively keep on removing the edge with the **Maximum Edge-Betweenness Centrality**, till we reach the *Required number of Connected Components aka Required Number of Clusters*. 

## Required Packages
	pandas 	    - Data Manipulation
	numpy 	    - Numeric Calculation
	matplotlib  - Graph Plotting
	networkx    - To make the Graph, Calculate Edge Betweenness Centrality
	sys 	    - Redirecting Standard Output
	os 	    - Opening File 
	copy 	    - Deep Copying of a Value

## Required Files
	CSV File "AAAI.csv" which stores the Information of the Papers Submitted

## Command Format
	python3 [script_name] 

## Execution
	Running Time around 40 seconds 

## Output

### GirvanNewmann_XX.txt
	Graph-Based Cluster with Threshold = 0.XX
### Proximity.csv
	Stores the Calculated Initial Proximity Matrix
