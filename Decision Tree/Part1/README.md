# DECISION TREE
In this folder, we build a Decision Tree on a Small DataSet for Car Sales.  
We have 4 features of 'price', 'maintenance', 'capacity', 'airbag' which take values in the form of strings/integers. Our *Target Class* 'profitable' takes a binary truth value, whether the car sale is profitable or not, given the features.  
The presence of strings renders a need to normalise the data into numbers. We **Numerise** the Data in such a way that we assign values starting from 0 till we cover all the unique possible outcomes of a feature. 
We use the criterion of ***entropy*** and ***gini index*** to grow a full tree and then use the testing data to report the accuracy of our trees compared to Scikit-Learn's models.

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
	Workbook "dataset for part 1.xlsx" in the same directory as the script.
	Sheet1 should be named 'Training Data', and
	Sheet2 should be named 'Testing Data'

## Command Format
	python3 [script_name] 

## Execution
	Running Time around 3-4 seconds
	Ignore Warnings if any

## Output
### Results.txt
	Numerised Training DataFrame
	Numerised Testing DataFrame
	Decision Tree using Information Gain (Self Learnt)
	Decision Tree using Gini Index (Self Learnt)
	Root Node Metrics Table for all the Decision Trees
	Result on Training Data for all the Decision Trees
	Result on Testing Data for all the Decision Trees
	Accuracy on Training Data for all the Decision Trees
	Accuracy on Testing Data for all the Decision Trees
	


