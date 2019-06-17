from anytree import Node, NodeMixin, RenderTree 
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import sys
import xlrd
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from numpy import log2
from numpy import inf 
np.set_printoptions(suppress=True, formatter={'float': '{: 0.6f}'.format}) 
pd.set_option('display.width', None)
eps = np.finfo(float).eps
err = 1e-8
Max_Depth = 1000000 

standard_output = sys.stdout 
sys.stdout = open("REPORT.txt", "w", encoding="utf-8")

Header = ["\nDECISION TREE using INFORMATION GAIN\n|", "\nDECISION TREE using GINI INDEX\n|"]

Aux = pd.read_excel( "dataset for part 1.xlsx" , sheet_name="Training Data" )
Mapping = {}
Rev_Mapping = {}
for feature in Aux.keys():
	Mapping[feature] = {}
	Rev_Mapping[feature] = {}
	Unique = np.unique(Aux[feature])
	for i in range(0 , len(Unique)):
		Mapping[feature][Unique[i]] = i 
		Rev_Mapping[feature][i] = Unique[i] 

def Numerise(DF):
	Len = len(DF)
	for feature in DF.keys():		
		for i in range(0,Len):
			DF[feature][i] = Mapping[feature][DF[feature][i]]
	return DF 

TrainDF = pd.read_excel( "dataset for part 1.xlsx" , sheet_name="Training Data" )
TrainDF = Numerise(TrainDF)		
Features = TrainDF.keys()[0:-1]
Class = TrainDF.keys()[-1]
Class_Val = np.unique(TrainDF[Class])							##Stores all the initially possible values of the Class
Num_of_Outcome = len(Class_Val)
Position = {}
for i in range(0, Num_of_Outcome):
	Position[Class_Val[i]] = i

def CalcNode(DF, Splitter):
	Class_Val_Count = [0]*Num_of_Outcome	
	for val in DF[Class].values:
		Class_Val_Count[Position[val]] +=1

	Num_of_Val = len(DF)
	Probabilty = np.array(Class_Val_Count)/Num_of_Val	

	if(Splitter == 0):
		Entropy = 0
		for p in Probabilty:
			Entropy -= p*log2(p+eps)
		if (Entropy < err ):
			Entropy = 0
		return Class_Val_Count, Entropy

	elif(Splitter == 1):
		Impurity = 1
		for p in Probabilty:
			Impurity -= p*p 
		return Class_Val_Count, Impurity

def CalcAttr(DF, Attribute, Splitter):
	Field = DF[Attribute].values	
	Attr_Val, Inv_Index, Attr_Val_Count = np.unique( Field , return_inverse=True, return_counts=True )
		
	Num_of_Val = len(DF)
	Weight_Val = Attr_Val_Count/Num_of_Val
		
	Len = len(Attr_Val)	
	Hash = np.zeros( shape=(Len, Num_of_Val), dtype=bool )
	for i in range(0, Num_of_Val):
		Hash[ Inv_Index[i] , i ] = True 	
	
	Value_Attr = 0
	for i in range(0, Len):
		df = DF[ Hash[i] ].reset_index(drop = True)
		Class_Val_Count, Value_Node = CalcNode(df, Splitter)
		Value_Attr += Weight_Val[i]*Value_Node
	
	return Value_Attr

def BestAttr(DF, Splitter):
	Value_Attr = []
	for attr in DF.keys()[:-1]:
		Value_Attr.append( CalcAttr(DF, attr, Splitter) )
	return DF.keys()[:-1][np.argmin(Value_Attr)]

def BuildTree(DF, Splitter, Parent ):
	flag = 0 													## To Store if Current Node is a Leaf
	if ( len(DF.keys()) == 1 ):									## No Attribute Left to Split on 
		flag = 1 
	if ( (Parent != None) and (Parent.depth == Max_Depth) ):	## Maximum Depth Reached
		flag = 2
	if ( CalcNode(DF, Splitter)[-1] == 0 ):								## Current Node is Perfectly Classified
		flag = 3	
	
	if( flag == 0 ):
		Split_Attr = BestAttr(DF, Splitter)
		Parent.split = Split_Attr 
		Attr_Val = np.unique( DF[Split_Attr] )
		for val in Attr_Val:
			Decision_Node = Node( Split_Attr, parent=Parent, value = val, split=None, child={}, outcome=None )
			Parent.child[val] = Decision_Node
			Leaf, Outcome = BuildTree( DF[ DF[Split_Attr]==val ].reset_index(drop=True), Splitter, Decision_Node )
			if ( Leaf == None ):
				Decision_Node.outcome = Outcome 
	else:
		Val, Count = np.unique( DF[Class] , return_counts=True )
		Outcome = Val[np.argmax(Count)]
		return None, Outcome

	return Parent, None

def Predict( Instance, Decision_Tree):
	if ( Decision_Tree.outcome != None ):
		return Decision_Tree.outcome
	return Predict( Instance, Decision_Tree.child[ Instance[Decision_Tree.split].values[0] ] ) 
	

def PrintTree( Node ):
	for pre, fill, node in RenderTree(Node):
		Print = pre + node.name.rstrip()	
		if ( node.is_root == False ):
			Print += " = " + str(Rev_Mapping[node.name][node.value]).rstrip()
		if ( node.is_leaf == True ):
			Print += " :: \"" + str(Rev_Mapping[Class][node.outcome]).rstrip() + "\""
		print(Print)

def PrintFrame( Heading, DF , w ):
	print( Heading.center(w))
	print("="*w)
	print(DF)
	print("")

PrintFrame("Numerised Training DataSet", TrainDF, 51)

TestDF = pd.read_excel( "dataset for part 1.xlsx" , sheet_name="Test Data" )
TestDF = Numerise(TestDF)
PrintFrame("Numerised Testing DataSet", TestDF, 51)

print("")

Root = []

Root.append(Node( name=Header[0], Parent=None, value=None, split=None, child={}, outcome=None))
Decision_Tree_Self_IG = BuildTree(TrainDF, 0, Root[0])[0]
PrintTree(Decision_Tree_Self_IG)
Self_Info = CalcNode(TrainDF, 0)[-1]
Self_InfoAttr =  CalcAttr(TrainDF, Decision_Tree_Self_IG.split, 0) 
Self_InfoGain = Self_Info - Self_InfoAttr
Metric_Self_IG = [Self_Info, Self_InfoAttr, Self_InfoGain]
print("\n")

Root.append(Node( name=Header[1], Parent=None, value=None, split=None, child={}, outcome=None))
Decision_Tree_Self_GI = BuildTree(TrainDF, 1, Root[1])[0]
PrintTree(Decision_Tree_Self_GI)
Self_Gini = CalcNode(TrainDF, 1)[-1] 
Self_GiniSplit = CalcAttr(TrainDF, Decision_Tree_Self_GI.split, 1)
Self_GiniDrop = Self_Gini - Self_GiniSplit
Metric_Self_GI = [Self_Gini, Self_GiniSplit, Self_GiniDrop]
print("\n")


Self_Train_IG = []
Self_Train_GI = []
Len_Train = len(TrainDF)
for i in range(0 , Len_Train ):
	Self_Train_IG.append( Predict( TrainDF.iloc[i:i+1], Decision_Tree_Self_IG ) )	
	Self_Train_GI.append( Predict( TrainDF.iloc[i:i+1], Decision_Tree_Self_GI ) )	

Self_Test_IG = []
Self_Test_GI = []
Len_Test = len(TestDF)
for i in range( 0, Len_Test):
	Self_Test_IG.append(Predict( TestDF.iloc[i:i+1], Decision_Tree_Self_IG ))
	Self_Test_GI.append(Predict( TestDF.iloc[i:i+1], Decision_Tree_Self_GI ))

Train_Data = TrainDF[Features].values.astype('int')
Train_Res = TrainDF[Class].values.astype('int')

Decision_Tree_SckLn_IG = DecisionTreeClassifier(criterion="entropy", max_depth=Max_Depth )
Decision_Tree_SckLn_IG = Decision_Tree_SckLn_IG.fit(Train_Data, Train_Res )
impurity =  Decision_Tree_SckLn_IG.tree_.impurity
samples = Decision_Tree_SckLn_IG.tree_.n_node_samples
SckLn_Info = impurity[0]
SckLn_InfoAttr =  (samples[1]*impurity[1] + samples[2]*impurity[2])/samples[0]
SckLn_InfoGain = SckLn_Info - SckLn_InfoAttr
Metric_SckLn_IG = [SckLn_Info, SckLn_InfoAttr, SckLn_InfoGain]

Decision_Tree_SckLn_GI = DecisionTreeClassifier(criterion="gini", max_depth=Max_Depth )
Decision_Tree_SckLn_GI = Decision_Tree_SckLn_GI.fit(Train_Data, Train_Res ) 
impurity =  Decision_Tree_SckLn_GI.tree_.impurity
samples = Decision_Tree_SckLn_GI.tree_.n_node_samples
SckLn_Gini = impurity[0]
SckLn_GiniSplit = (samples[1]*impurity[1] + samples[2]*impurity[2])/samples[0]
SckLn_GiniDrop = SckLn_Gini - SckLn_GiniSplit
Metric_SckLn_GI = [SckLn_Gini, SckLn_GiniSplit, SckLn_GiniDrop]

frame = { "Self_InfoGain":Metric_Self_IG, "SckLn_InfoGain":Metric_SckLn_IG, "Self_GiniIndex":Metric_Self_GI, "SckLn_GiniIndex":Metric_SckLn_GI}
Index = [ "Root Impurity", "Attribute Impurity" , "Impurity Reduction"]
Metric_DF = pd.DataFrame(data=frame, index=Index)
PrintFrame( "Metrics at Root Node", Metric_DF, 82)
print("")

SckLn_Train_IG = Decision_Tree_SckLn_IG.predict(Train_Data)
SckLn_Train_GI = Decision_Tree_SckLn_GI.predict(Train_Data)

Test_Data = TestDF[Features].values.astype('int')
Test_Res = TestDF[Class].values.astype('int')
SckLn_Test_IG = Decision_Tree_SckLn_IG.predict(Test_Data)
SckLn_Test_GI = Decision_Tree_SckLn_GI.predict(Test_Data)

frame = { "Self_IG":Self_Train_IG, "SckLn_IG":SckLn_Train_IG, "Self_GI":Self_Train_GI, "SckLn_GI":SckLn_Train_GI, "Actual":Train_Res }
Res_TrainDF = pd.DataFrame( data=frame )
Res_TrainDF = pd.DataFrame.replace(Res_TrainDF, Rev_Mapping[Class] )
PrintFrame("Result on Training Dataset", Res_TrainDF, 42)

frame = { "Self_IG":Self_Test_IG, "SckLn_IG":SckLn_Test_IG, "Self_GI":Self_Test_GI, "SckLn_GI":SckLn_Test_GI, "Actual":Test_Res }
Res_TestDF = pd.DataFrame( data=frame )
Res_TestDF = pd.DataFrame.replace(Res_TestDF, Rev_Mapping[Class] )
PrintFrame("Result on Testing Dataset", Res_TestDF, 42)

print("")

pd.options.display.float_format = '{:.2f}%'.format

Acc_Self_Train_IG = metrics.accuracy_score(Self_Train_IG, Train_Res)*100
Acc_SckLn_Train_IG = metrics.accuracy_score(SckLn_Train_IG, Train_Res)*100
Acc_Self_Train_GI = metrics.accuracy_score(Self_Train_GI, Train_Res)*100
Acc_SckLn_Train_GI = metrics.accuracy_score(SckLn_Train_GI, Train_Res)*100
frame = { "Self_IG":[Acc_Self_Train_IG], "SckLn_IG":[Acc_SckLn_Train_IG], "Self_GI":[Acc_Self_Train_GI], "SckLn_GI":[Acc_SckLn_Train_GI] }
Acc_TrainDF = pd.DataFrame( data=frame )
PrintFrame("Accuracy on Training Dataset", Acc_TrainDF, 39)

Acc_Self_Test_IG = metrics.accuracy_score(Self_Test_IG, Test_Res)*100
Acc_SckLn_Test_IG = metrics.accuracy_score(SckLn_Test_IG, Test_Res)*100
Acc_Self_Test_GI = metrics.accuracy_score(Self_Test_GI, Test_Res)*100
Acc_SckLn_Test_GI = metrics.accuracy_score(SckLn_Test_GI, Test_Res)*100
frame = { "Self_IG":[Acc_Self_Test_IG], "SckLn_IG":[Acc_SckLn_Test_IG], "Self_GI":[Acc_Self_Test_GI], "SckLn_GI":[Acc_SckLn_Test_GI] }
Acc_TestDF = pd.DataFrame( data=frame )
PrintFrame("Accuracy on Testing Dataset", Acc_TestDF, 39)

sys.stdout = standard_output