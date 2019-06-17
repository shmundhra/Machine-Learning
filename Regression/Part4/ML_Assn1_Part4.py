import os 
import sys
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
np.set_printoptions(suppress=True) 
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
pd.set_option('display.width', None)
#np.seterr(all='ignore')
standard_output = sys.stdout 

# Part 1a ) 
## Generation of synthetic dataset
def GenerateDataset( M ) :  
    from numpy import pi  
    MYU = 0 
    SIGMA = 0.3
    X  = np.random.rand( M , 1 )                    # Column Vector X = np.random.uniform( 0.0, 1.0, M )
    Y0 = np.sin( 2*pi*X )
    E  = SIGMA * np.random.rand( M , 1 ) + MYU      # Column Vector E = np.random.normal( MYU, SIGMA, M )
    Y  = Y0 + E
    Dataset = np.hstack((X,Y))      
    '''
    frame = { "X Coordinates" : Dataset[: , 0] , "Y Coordinates" : Dataset[: , 1] }
    DF = pd.DataFrame(data=frame , index = [ i+1 for i in range(0 , Dataset.shape[0])])
    print("DATASET".center(32) )
    print('-'*32 ) 
    print( DF )     
    '''
    return Dataset  

# Part 1b ) 
## Splitting dataset into Training and Test
def Split( Dataset , Training_Fraction ):


    Shuffle = np.array(Dataset)    
    np.random.shuffle(Shuffle)                      #Multi-dimensional arrays are only shuffled along the first axis:
    Split = round(Training_Fraction*M)
    Training, Testing = Shuffle[ :Split , :] , Shuffle[ Split: , :]

    Training = Training[Training[ : , 0 ].argsort()]
    Train_X = np.array(Training[ : , 0 ])
    Train_X.shape =  ( 1 , Train_X.shape[0] ) 
    Train_Y = np.array(Training[ : , 1 ])
    Train_Y.shape =  ( 1 , Train_Y.shape[0] )

    Testing = Testing[Testing[ : , 0 ].argsort()]
    Test_X = np.array(Testing[ : , 0 ])
    Test_X.shape =  ( 1 , Test_X.shape[0] ) 
    Test_Y = np.array(Testing[ : , 1 ])   
    Test_Y.shape =  ( 1 , Test_Y.shape[0] )  
    '''
    frame = { "X Coordinates" : Train_X[0] , "Y Coordinates" : Train_Y[0] }
    DF = pd.DataFrame(data = frame , index = [ i+1 for i in range(0 , Train_X.shape[1] ) ] )
    print("\n" , "TRAINING SET".center(32) )
    print('-'*32 ) 
    print( DF )  

    frame = { "X Coordinates" : Test_X[0] , "Y Coordinates" : Test_Y[0] }
    DF = pd.DataFrame(data = frame , index = [ i+1 for i in range(0 , Test_X.shape[1] ) ] ) 
    print("\n" , "TESTING SET".center(32) ) 
    print('-'*32 ) 
    print( DF )
    '''
    return Train_X, Train_Y, Test_X, Test_Y 

def designMatrix( Dataset , Degree ):
    Power = np.array( [ p for p in range(0 , Degree+1) ] )
    Power.shape = ( Degree+1 , 1 )  
    Matrix = np.repeat( Dataset , Degree+1 , axis=0 )
    Matrix = np.power( Matrix , Power )
    return Matrix 
 
def h (ThetaT, X ): 
    Estimate = ThetaT.dot(X)    
    return Estimate                 #Estimate is ( 1 , M )

def error(ThetaT, X, Y):
    Error = h(ThetaT, X) - Y 
    return Error                    #Error is ( 1 , M ) 

def partial_Diff( ThetaT, X, Y, index):
    Error = error(ThetaT, X, Y)

    if ( index == 0 ):
        Partial_Diff = 2*Error

    elif ( index == 1 ):
        Partial_Diff = np.sign(Error)

    elif ( index == 2 ):
        Error = np.array( Error , dtype=np.float64 )
        Partial_Diff = np.power( Error , 3 )    
        Partial_Diff *= 4  

    return Partial_Diff 

def cost( ThetaT, X, Y, index ):
    M = Y.shape[1]
    Factor = float(0.5 / M )    
    Error = error( ThetaT, X, Y)

    if ( index == 0 ):
        Cost = (Error).dot(Error.T) 
        Cost = Cost[0][0]        
    
    elif ( index == 1 ):
        Sgn = np.sign(Error)
        Cost = (Error).dot(Sgn.T)
        Cost = Cost[0][0]       

    elif ( index == 2 ): 
        Cost = (Error).dot(Error.T)
        Cost = np.array(Cost , dtype=np.float64)
        Cost = np.power(Cost , 2 )
        Cost = Cost[0][0]        

    Cost = Factor*Cost
    return Cost

def descent( ThetaT, X, Y, index ):
    M = Y.shape[1] 
    Factor = float(0.5/M) 

    Partial_Diff = partial_Diff( ThetaT, X, Y , index ) 
    Descent = Factor * Partial_Diff.dot(X.T)                     #Descent is ( 1 , N+1 )

    return Descent  

COST_FUNCTION = [ "Square" , "Absolute" , "Biquadratic" ]
LEARNING_RATES = [0.025, 0.05, 0.1, 0.2, 0.5]
ITERATIONS = 3000
DATASET_SIZE = [10 ,100, 1000, 10000 ]
Training_Fraction = 0.8 
MAX_Degree = 9 
DEGREE = [ i for i in range(1 , MAX_Degree+1) ]
LR_Training_Cost, LR_Testing_Cost, LR_Final_ThetaT = [], [], []

os.makedirs("RMS Error vs Learning Rate", exist_ok=True)
os.makedirs("RMS Error vs Learning Rate/DatasetSize_Variation", exist_ok=True)
os.makedirs("RMS Error vs Learning Rate/CurveDegree_Variation", exist_ok=True)
sys.stdout = open("Results.txt", "w")

for LR in LEARNING_RATES : 

    M_Training_Cost, M_Testing_Cost, M_Final_ThetaT = [], [], []
    for M in DATASET_SIZE:
        
        os.makedirs("RMS Error vs Learning Rate/CurveDegree_Variation/DatasetSize_"+str(M), exist_ok=True)
        Dataset = GenerateDataset(M)    
        Train_X, Train_Y, Test_X, Test_Y = Split( Dataset , Training_Fraction )   

        CF_Training_Cost, CF_Testing_Cost, CF_Final_ThetaT = [], [], []
        for CF in range( 0 , len(COST_FUNCTION) ):
                    
            N_Training_Cost, N_Testing_Cost, N_Final_ThetaT = [], [], []   
            for N in DEGREE :  
                os.makedirs("RMS Error vs Learning Rate/DatasetSize_Variation/CurveDegree_"+str(N), exist_ok=True)              
                ThetaT = (0.00001)*np.random.randn( 1 , N+1 )
                Training_X = designMatrix( Train_X, N ) 
                Testing_X = designMatrix( Test_X, N )
                 
                CostVec = []
                for i in range(0 , ITERATIONS): 
                    CostVec.append( cost(ThetaT, Training_X, Train_Y , CF ) )
                    ThetaT = ThetaT - LR*descent(ThetaT, Training_X, Train_Y, CF )

                N_Training_Cost.append( cost(ThetaT, Training_X, Train_Y , 0 ) )
                N_Testing_Cost.append( cost(ThetaT, Testing_X, Test_Y, 0 ) )
                N_Final_ThetaT.append( ThetaT[0] ) 

            CF_Training_Cost.append(N_Training_Cost)
            CF_Testing_Cost.append(N_Testing_Cost)
            CF_Final_ThetaT.append(N_Final_ThetaT)

        M_Training_Cost.append(CF_Training_Cost)
        M_Testing_Cost.append(CF_Testing_Cost)
        M_Final_ThetaT.append(CF_Final_ThetaT)

    LR_Training_Cost.append(M_Training_Cost)
    LR_Testing_Cost.append(M_Testing_Cost)
    LR_Final_ThetaT.append(M_Final_ThetaT)   

for m in range( 0 , len(DATASET_SIZE)) :
    M = DATASET_SIZE[m]
    for n in range( 0 , len(DEGREE) ):
        N = DEGREE[n]
        plt.figure(figsize=(12, 8))
        plt.xlabel("Learning Rate")
        plt.ylabel("Root Mean Square Error")  
        for cf in range( 0 , len(COST_FUNCTION) ):            
            CF = COST_FUNCTION[cf] 
            Temp = np.array(LR_Testing_Cost)
            plt.plot(LEARNING_RATES , Temp[: , m , cf , n ] , '.-' , label=CF )
        plt.title("VARIATION of RMSE of Cost Functions with Learning Rate (M=" + str(M) + ", N=" + str(N) + ")" )
        plt.legend() 
        plt.grid(True)
        os.chdir("RMS Error vs Learning Rate/DatasetSize_Variation/CurveDegree_"+str(N))        
        plt.savefig( ("M_" + str(M) + "_N_" + str(N) ) )
        os.chdir("../../..")
        os.chdir("RMS Error vs Learning Rate/CurveDegree_Variation/DatasetSize_"+str(M))
        plt.savefig( ("M_" + str(M) + "_N_" + str(N) ) )
        os.chdir("../../..")
        #plt.show()
        plt.close()

for lr in range( 0 , len(LEARNING_RATES)) : 
    LR = LEARNING_RATES[lr]
    for m in range( 0 , len(DATASET_SIZE)) :
        M = DATASET_SIZE[m]
        for cf in range( 0 , len(COST_FUNCTION) ):            
            CF = COST_FUNCTION[cf]

            Columns = [ "Theta"+str(c) for c in range(0,MAX_Degree+1) ]
            Index = [ "Deg"+str(d) for d in DEGREE]
            DF = pd.DataFrame( LR_Final_ThetaT[lr][m][cf] , index = Index , columns = Columns ) 
            print("")
            print(("PARAMETERS LEARNED for LR=" + str(LR) + ", M=" + str(M) + ", CF=Mean" + CF ).center(104) )
            print("-"*104)
            print(DF)

            frame = { "Features": [ d+1 for d in DEGREE ] , "Training Error": LR_Training_Cost[lr][m][cf] , "Testing Error": LR_Testing_Cost[lr][m][cf] , "Difference": [LR_Training_Cost[lr][m][cf][i]-LR_Testing_Cost[lr][m][cf][i] for i in range(0, MAX_Degree)] }
            DF = pd.DataFrame(frame)
            DF.set_index("Features" , inplace=True )
            print("")  
            print( "VARIATION OF ERRORS WITH FEATURES".center(51) ) 
            print("-"*51 )
            print(DF)   

sys.stdout = standard_output