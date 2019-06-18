# CLUSTERING EVALUATION

In this folder, we aim to evaluate the sets of clusters we obtain using SinglyLinked, CompletelyLinked and GirvanNewman Algorithms( with different thresholds). We use the ***set of High-Level Keyword(s)*** as the *gold standard* for evaluation and use **NORMALIZED MUTUAL INFORMATION** as the Metric. 

## Required Packages
	pandas 	    - Data Manipulation
	numpy 	    - Numeric Calculation
	matplotlib  - Graph Plotting
	anytree     - Storing the Dendogram 
	networkx    - To make the Graph, Calculate Edge Betweenness Centrality
	sys 	    - Redirecting Standard Output
	os 	    - Opening File 
	copy 	    - Deep Copying of a Value

## Required Files
	CSV File "AAAI.csv" which stores the Information of the Papers Submitted

## Command Format
	python3 [script_name] 

## Execution
	Running Time around 75 seconds 
	Ignore Warnings if any

## Output
	
### Results.txt 
	Stores the Clusters formed by Single and Complete Linkage, and GirvanNewman with different thresholds
### SingleLinkage.txt
	Stores the Dendogram for Singly Linked Clustering  
### CompleteLinkage.txt 
	Stores the Dendogram for Completely Linked Clustering 
### NMI_Values.csv 
	Stores the NMI Values for the Different Generated Clusters
### Proximity.csv
	Stores the Calculated Initial Proximity Matrix
