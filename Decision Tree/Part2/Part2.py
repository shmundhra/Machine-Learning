from anytree import Node, RenderTree 
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import sys
import os 
import copy
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from numpy import log2
np.set_printoptions(suppress=True, formatter={'float': '{: 0.2f}%'.format}, threshold=np.nan)
pd.set_option('display.width', None)
eps = np.finfo(float).eps
err = 1e-8
standard_output = sys.stdout
criterion = {0:"entropy", 1:"gini"}
Split_Criteria = {0:"INFORMATION GAIN", 1:"GINI INDEX"}

if( sys.argv[1] == '0' or sys.argv[1] == '1' ):
	Splitter = int(sys.argv[1])
else:
	print("Command Format: python3 [script_name] [criterion_index]")
	exit

Train_Label = [0]
Train_Size = 0
with open("trainlabel.txt") as f:
	for line in f:
		Train_Size+=1
		Train_Label.append( int(line.strip()) )
f.close()

Mapping = {}
Rev_Mapping = {}
F = 0 
with open("words.txt") as f:
	for line in f:
		F+=1
		word = line.strip()
		Mapping[word] = F
		Rev_Mapping[F] = word
f.close()

Class = "TOPIC"	
Topic = {1:"alt.atheism" , 2:"comp.graphics"}
Class_Val = np.unique(Train_Label[1:])
Num_of_Outcome = len(Class_Val)
Postion_Outcome = {}
for i in range( Num_of_Outcome ):
	Postion_Outcome[Class_Val[i]] = i

Columns = [ Rev_Mapping[i] for i in range(1, F+1) ]
Columns.append(Class)

def CalcNode(Class_Values, Splitter):
	Class_Val_Count = [0]*Num_of_Outcome	
	for val in Class_Values:
		Class_Val_Count[Postion_Outcome[val]] +=1
	
	Probabilty = np.array(Class_Val_Count)/len(Class_Values)	

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
	Attr_Val, Inv_Index, Attr_Val_Count = np.unique( DF[Attribute].values , return_inverse=True, return_counts=True )
	Weight_Val = Attr_Val_Count/len(DF)	
	
	Hash = np.zeros( shape=( len(Attr_Val), len(DF) ), dtype=bool )
	for i in range( len(DF) ):
		Hash[ Inv_Index[i] , i ] = True 	
	
	Value_Attr = 0
	for i in range( len(Attr_Val)	):
		df = DF[ Hash[i] ]
		Class_Val_Count, Value_Node = CalcNode( df[Class].values, Splitter)
		Value_Attr += Weight_Val[i]*Value_Node
	
	return Value_Attr

def BestAttr(DF, Splitter):
	Attr_Split_Value = []
	Attributes = DF.keys()[:-1]	
	for attr in Attributes:
		frame = { attr:DF[attr].values, Class:DF[Class].values }
		df = pd.DataFrame(data=frame)
		Attr_Split_Value.append( CalcAttr(df, attr, Splitter) )		
	return Attributes[np.argmin(Attr_Split_Value)]
	
def BuildTree(Parent, Splitter, Max_Depth ):

	if not Parent.is_leaf:
		for subtree in Parent.children :
			BuildTree( subtree, Splitter, Max_Depth )
		return Parent, None

	flag = 0 													## To Store if Current Node is a Leaf
	if ( len(Parent.df.keys()) == 1 ):									## No Attribute Left to Split on 
		flag = 1 
	if ( not(Parent.is_root) and (Parent.depth == Max_Depth) ):	## Maximum Depth Reached
		flag = 2
	if ( CalcNode(Parent.df[Class].values, Splitter)[-1] == 0 ):		## Current Node is Perfectly Classified
		flag = 3	
	
	if( flag == 0 ):
		Split_Attr = BestAttr(Parent.df, Splitter)
		Parent.split = Split_Attr 
		Attr_Val, Inv_Index = np.unique( Parent.df[Split_Attr], return_inverse=True )

		Hash = np.zeros( shape=( len(Attr_Val), len(Parent.df) ), dtype=bool )
		for i in range( len(Parent.df) ):
			Hash[ Inv_Index[i] , i ] = True

		for i in range( len(Attr_Val) ):
			Decision_Node = Node( Split_Attr, parent=Parent, df=Parent.df[ Hash[i] ], value = Attr_Val[i], split=None, child={}, outcome=None )
			Parent.child[Attr_Val[i]] = Decision_Node
			Leaf, Outcome = BuildTree( Decision_Node, Splitter, Max_Depth  )
			if ( Leaf == None ):
				Decision_Node.outcome = Outcome
				
	else:
		Val, Count = np.unique( Parent.df[Class] , return_counts=True )
		Outcome = Val[np.argmax(Count)]
		return None, Outcome

	return Parent, None

def Predict( Instance, Decision_Tree ):
	if ( Decision_Tree.is_leaf ):
		return Decision_Tree.outcome
	return Predict( Instance, Decision_Tree.child[ Instance[Decision_Tree.split].values[0] ] ) 

def PrintTree( Node ):
	for pre, fill, node in RenderTree(Node):
		Print = pre + str(node.name).strip()
		if not node.is_root :
			Print += " = " + str(int(node.value))
		if node.is_leaf :
			Print += " :: \"" + Topic[node.outcome] + "\""
		print(Print)

#________________________________________________________________________________________________________
Train_DataSet = np.zeros( shape=(Train_Size +1, F+1 +1 ), dtype=np.int8 )
with open("traindata.txt") as f:
	for line in f:
		cell = line.strip().split()
		Train_DataSet[ int(cell[0]), int(cell[1]) ] = 1		
f.close()

for i in range( len(Train_Label) ):
	Train_DataSet[i,-1] = Train_Label[i]

Train_DF = pd.DataFrame(data=Train_DataSet[1:,1:], columns=Columns )
Train_DF.index+=1

Test_Label = [0]
Test_Size = 0
with open("testlabel.txt") as f:
	for line in f:
		Test_Size +=1 
		Test_Label.append( int(line.rstrip()) )
f.close()

Test_DataSet =  np.zeros( shape=(Test_Size +1, F+1 +1 ), dtype=np.int8 )
with open("testdata.txt") as f:
	for line in f:
		cell = line.strip().split()
		Test_DataSet[ int(cell[0]), int(cell[1]) ] = 1	
f.close()

for i in range( len(Test_Label) ):
	Test_DataSet[i,-1] = Test_Label[i]

Test_DF = pd.DataFrame( data=Test_DataSet[1:,1:] , columns=Columns )
Test_DF.index +=1 

Height = 0
Root = []
Acc_Self_Train, Acc_SckLn_Train, Acc_Self_Test, Acc_SckLn_Test = [], [], [], []
Res_Self_Train, Res_SckLn_Train, Res_Self_Test, Res_SckLn_Test = [], [], [], []
##______________________________________________________________________________________________________
os.makedirs("DecisionTrees_"+criterion[Splitter], exist_ok=True )
os.chdir("DecisionTrees_"+criterion[Splitter])
Node_Name = "\nDECISION TREE using " + Split_Criteria[Splitter] + " with Max_Depth = "
Decision_Tree_Self = Node( Node_Name, Parent=None, df=Train_DF, value=None, split=None, child={}, outcome=None)
for depth in range(1, F+1):	
	Decision_Tree_Self = BuildTree( Decision_Tree_Self, Splitter , depth )[0]
	Root.append( copy.deepcopy(Decision_Tree_Self) )
	Root[-1].name += str(depth) + "\n|"		 	
	print( "Decision Tree with MaxDepth %d learnt..." % depth )
	sys.stdout = open( ( "DTree_MaxDepth"+str(depth)+".txt" ) , "w", encoding="utf-8" )
	PrintTree(Root[-1])	
	sys.stdout = standard_output	

	Train_Data = Train_DataSet[1:,1:-1] 
	Train_Res = Train_DataSet[1:,-1]
	
	Self_Train = []
	for i in range( len(Train_DF) ):
		Self_Train.append( Predict(Train_DF.iloc[i:i+1], Root[-1]) )
	Res_Self_Train.append(Self_Train)
	Acc_Self_Train.append( metrics.accuracy_score(Self_Train, Train_Res)*100 )
	
	Test_Data = Test_DataSet[1:,1:-1]
	Test_Res = Test_DataSet[1:,-1]

	Self_Test = []	
	for i in range( len(Test_DF) ):
		Self_Test.append( Predict(Test_DF.iloc[i:i+1], Root[-1]) )
	Res_Self_Test.append( Self_Test )
	Acc_Self_Test.append( metrics.accuracy_score(Self_Test, Test_Res)*100 )
	
	Train_Data = Train_DataSet[1:,1:-1] 
	Train_Res = Train_DataSet[1:,-1]
	Decision_Tree_SckLn = DecisionTreeClassifier(criterion=criterion[Splitter] , max_depth=depth)
	Decision_Tree_SckLn = Decision_Tree_SckLn.fit(Train_Data, Train_Res)
	
	SckLn_Train = Decision_Tree_SckLn.predict(Train_Data)	
	Res_SckLn_Train.append( SckLn_Train )	
	Acc_SckLn_Train.append( metrics.accuracy_score(SckLn_Train , Train_Res)*100 )
	
	Test_Data = Test_DataSet[1:,1:-1]
	Test_Res = Test_DataSet[1:,-1]
	SckLn_Test = Decision_Tree_SckLn.predict(Test_Data)
	Res_SckLn_Test.append( SckLn_Test )
	Acc_SckLn_Test.append( metrics.accuracy_score(SckLn_Test, Test_Res)*100 )

	if ( Root[-1].height == Height ):
		break
	Height = Root[-1].height
os.chdir("..")

BestTree_Self = np.argmax(Acc_Self_Test)
print( "The Depth of the Tree which maximises Testing Accuracy is : %d" % (BestTree_Self+1) )

frame = {"Self_Train":Res_Self_Train[BestTree_Self] , "SckLn_Train":Res_SckLn_Train[BestTree_Self] , "Actual_Label":Train_Res}
frame.update( { "Self_Misclassification"  : abs( np.array(frame["Actual_Label"]) - np.array(frame["Self_Train"]  ) ) } )
frame.update( { "SckLn_Misclassification" : abs( np.array(frame["Actual_Label"]) - np.array(frame["SckLn_Train"] ) ) } )
frame.update( { "SckLn <> Self" : abs(frame["SckLn_Misclassification"]-frame["Self_Misclassification"]) } )
BestOut_TrainDF = pd.DataFrame(data=frame)

frame = {"Self_Test":Res_Self_Test[BestTree_Self] , "SckLn_Test":Res_SckLn_Test[BestTree_Self] , "Actual_Label":Test_Res}
frame.update( { "Self_Misclassification"  : abs( np.array(frame["Actual_Label"]) - np.array(frame["Self_Test"]  ) ) } )
frame.update( { "SckLn_Misclassification" : abs( np.array(frame["Actual_Label"]) - np.array(frame["SckLn_Test"] ) ) } )
frame.update( { "SckLn <> Self" : abs(frame["SckLn_Misclassification"] -frame["Self_Misclassification"]) } )
BestOut_TestDF = pd.DataFrame(data=frame)

Depths = [ d+1 for d in range( len(Acc_Self_Train) ) ]

frame = {"Max_Depth": Depths, "Acc_Self_Train":Acc_Self_Train, "Acc_SckLn_Train":Acc_SckLn_Train, "Acc_Self_Test":Acc_Self_Test, "Acc_SckLn_Test":Acc_SckLn_Test}
Acc_DF = pd.DataFrame(data=frame)
Acc_DF = Acc_DF.set_index("Max_Depth", inplace=False)
Acc_DF = Acc_DF[Acc_DF.keys()].apply(lambda x: round(x,2).astype(str)+"%" )

os.makedirs("DataFrames_"+criterion[Splitter], exist_ok=True)
os.chdir("DataFrames_"+criterion[Splitter])
Train_DF.to_csv(r"train_dataset.csv")
Test_DF.to_csv(r"test_dataset.csv")
BestOut_TrainDF.to_csv(r"BestTreeOutcome_TrainingData.csv")
BestOut_TestDF.to_csv(r"BestTreeOutcome_TestingData.csv")
Acc_DF.to_csv(r"Accuracy_Percentage.csv")
os.chdir("..")

os.makedirs("Plots_"+criterion[Splitter], exist_ok=True )
os.chdir("Plots_"+criterion[Splitter])

fig = plt.figure(figsize=(10,6))
ax = fig.gca()
ax.set_xticks(np.arange(1, len(Depths)+1, 1))
plt.xlabel("Max_Depth")
plt.ylabel("Accuracy (in %)")
plt.plot( np.array(Depths).astype('int'), Acc_SckLn_Train, 'o-' , label="Scikit_Learn")
plt.plot( np.array(Depths).astype('int'), Acc_Self_Train, 'o-', label="Self_Learn" )
plt.title("Accuracy vs Depth (Training Dataset)")
plt.legend()
plt.grid(True)
plt.savefig("AccVSDepth_Training")
#plt.show()
plt.close()

fig = plt.figure(figsize=(10,6))
ax = fig.gca()
ax.set_xticks(np.arange(1, len(Depths)+1, 1))
plt.xlabel("Max_Depth")
plt.ylabel("Accuracy (in %)")
plt.plot( np.array(Depths).astype('int'), Acc_SckLn_Test, 'o-' , label="Scikit_Learn")
plt.plot( np.array(Depths).astype('int'), Acc_Self_Test, 'o-', label="Self_Learn" )
plt.title("Accuracy vs Depth (Testing Dataset)")
plt.legend()
plt.grid(True)
plt.savefig("AccVSDepth_Testing")
#plt.show()
plt.close()

fig = plt.figure(figsize=(10,6))
ax = fig.gca()
ax.set_xticks(np.arange(1, len(Depths)+1, 1))
plt.xlabel("Max_Depth")
plt.ylabel("Accuracy (in %)")
plt.plot( np.array(Depths).astype('int'), Acc_Self_Train, 'o-' , label="Self_Train")
plt.plot( np.array(Depths).astype('int'), Acc_Self_Test, 'o-' , label="Self_Test")
plt.title("Learning Curve (Self)")
plt.legend()
plt.grid(True)
plt.savefig("LearnCurve_Self")
#plt.show()
plt.close()

fig = plt.figure(figsize=(10,6))
ax = fig.gca()
ax.set_xticks(np.arange(1, len(Depths)+1, 1))
plt.xlabel("Max_Depth")
plt.ylabel("Accuracy (in %)")
plt.plot( np.array(Depths).astype('int'), Acc_SckLn_Train, 'o-' , label="SckLn_Train")
plt.plot( np.array(Depths).astype('int'), Acc_SckLn_Test, 'o-' , label="SckLn_Test")
plt.title("Learning Curve (SckLn)")
plt.legend()
plt.grid(True)
plt.savefig("LearnCurve_SckLn")
#plt.show()
plt.close()

os.chdir("..")