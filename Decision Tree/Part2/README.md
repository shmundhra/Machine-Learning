# DECISION TREE
In this folder, we build a Decision Tree on a DataSet of the Bag of Words Model.  
We have 3566 words taken from articles related to **alt.atheism** and **comp.graphics** which are assigned numeric keys in *'words.txt'*. These act as features in our dataset. We assign a numeric value of **1 for alt.atheism** and **2 for comp.graphics** in our labeling set to the different news articles.  
We use the criterion of ***entropy*** and ***gini index*** to grow trees progressively from a Depth of 1 till when the Depth Saturates and then use the testing data to report the tree with the **Highest Testing Accuracy**.  
We then **compare** our model results with **Scikit-Learn's models**.

## Required Packages
	pandas 	    - Data Manipulation
	numpy 	    - Numeric Calculation
	matplotlib  - Graph Plotting
	sklearn     - ML Library Decision Tree
	anytree     - Self Decision Tree Implementation
	xlrd 	    - Reading from Excel File
	sys 	    - Redirecting Standard Output
	os 	    - Opening File 
	copy 	    - Deep Copying of a Value

## Required Files
	'traindata.txt'
	'trainlabel.txt'
	'testdata.txt'
	'testlabel.txt'
	'words.txt'
	All Files should be present in the same working directory

## Command Format
	python3 [script_name] [criterion]  
	Use criterion = 0 for "entropy"
	Use criterion = 1 for "gini"

## Execution
	Running Time around 400 seconds

## Output

### DecisionTrees_entropy
	Contains Self-Learnt DecisionTrees from Depth1 to Max_Depth with learning criterion as "entropy"

### DecisionTrees_gini
	Contains Self-Learnt DecisionTrees from Depth1 to Max_Depth with learning criterion as "gini"

### DataFrames_entropy
	Accuracy_Percentage.csv 		- Accuracy of Learnt Trees (entropy) at all Depths
	BestTreeOutcome_TrainingData.csv 	- Results of the Best Tree (entropy) on Training Data 
	BestTreeOutcome_TestingData.csv 	- Results of the Best Tree (entropy) on Testing Data
	train_dataset.csv 			- Training Dataset reproduced in table format
	test_dataset.csv 			- Testing Dataset reproduced in table format 

### DataFrames_gini
	Accuracy_Percentage.csv 		- Accuracy of Learnt Trees (gini) at all Depths
	BestTreeOutcome_TrainingData.csv 	- Results of the Best Tree (gini) on Training Data 
	BestTreeOutcome_TestingData.csv 	- Results of the Best Tree (gini) on Testing Data
	train_dataset.csv 			- Training Dataset reproduced in table format
	test_dataset.csv 			- Testing Dataset reproduced in table format

### Plots_entropy
	AccVSDepth_Training	-Plot of Training Accuracy of Learnt Trees (self and scikit) vs Max_Depth of Tree (entropy)
	AccVSDepth_Testing	-Plot of Testing  Accuracy of Learnt Trees (self and scikit) vs Max_Depth of Tree (entropy) 
	LearnCurve_Self		-Plot of Training Accuracy and Testing Accuracy vs Max_Depth of Tree (self - entropy)
	LearnCurve_SckLn	-Plot of Training Accuracy and Testing Accuracy vs Max_Depth of Tree (scikit - entropy)

### Plots_gini
	AccVSDepth_Training	-Plot of Training Accuracy of Learnt Trees (self and scikit) vs Max_Depth of Tree (gini)
	AccVSDepth_Testing	-Plot of Testing  Accuracy of Learnt Trees (self and scikit) vs Max_Depth of Tree (gini) 
	LearnCurve_Self		-Plot of Training Accuracy and Testing Accuracy vs Max_Depth of Tree (self - gini)
	LearnCurve_SckLn	-Plot of Training Accuracy and Testing Accuracy vs Max_Depth of Tree (scikit - gini)
	


