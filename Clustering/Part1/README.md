# HIERARCHICAL AGGLOMERATIVE CLUSTERING 

In this folder, we aim to create a Clustering of the given Dataset of **Submitted Papers in the AAAI**, in a Bottom-Up Fashion using a Distance-Based Clustering Method. The Metric of Similarity between two Papers is the **Jaccard Coefficient** applied on the ***set of Topics*** which each Paper caters to. 
We have used *Single Linkage* and *Complete Linkage* to generate two Clusterings. 


## DATA STRUCTURES and ALGORITHM 

We use a **Proximity Matrix**, which stores the Proximity 
between each pairs of *Paper Clusters*.  
We use a **Closest Matrix**, which stores for each Cluster the Index of the Cluster *Closest* to it, and the value of that *Proximity*.  
We then **choose the Two Clusters which have the Maximum Proximity and Merge them into each other.**  
While merging Cluster(I) and Cluster(II), we make each reference to Cluster(I) in the **Closest Matrix** to point now to Cluster(II), and set the Cluster(I) corresponding calues to *nan*.  
The updation of the **Proximity Matrix** depends on the type of Linkage.  

### MIN or SINGLE LINKAGE  
In Single Linkage, Proximity is defined as the **Maximum of Similarity between any two pair of points in two corresponding clusters.** Thus when we are merging two Clusters into one, the row corresponding to that cluster shall have the **Max Value of the two rows at each cell**.

### MAX or COMPLETE LINKAGE  
In Complete Linkage, Proximity is defined as the **Minimum of Similarity between any two pair of points in two corresponding clusters.** Thus when we are merging two Clusters into one, the row corresponding to that cluster shall have the **Min Value of the two rows at each cell**.

After we have updated the **Proximity Matrix**, we update the **Closest Matrix** and then move on to the next iteration.

## Required Packages
	pandas 	    - Data Manipulation
	numpy 	    - Numeric Calculation
	matplotlib  - Graph Plotting
	anytree     - Storing the Dendogram
	sys 	    - Redirecting Standard Output
	os 	    - Opening File 
	copy 	    - Deep Copying of a Value

## Required Files
	CSV File "AAAI.csv" which stores the Information of the Papers Submitted

## Command Format
	python3 [script_name] 

## Execution
	Running Time around 1 second

## Output

### SingleLinkage.txt
	Singly Linked Cluster Information 
### CompleteLinkage.txt 
	Completely Linked Cluster Information 
### Proximity.csv 
	Stores the Calculated Initial Proximity Matrix
