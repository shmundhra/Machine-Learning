import os 
import sys
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import itertools
np.set_printoptions(suppress=True) 
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
pd.set_option('display.width', None)
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
    return Dataset,X.T,Y.T

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

def h ( ThetaT, X ): 
    Estimate = ThetaT.dot(X)    
    return Estimate                 #Estimate is ( 1 , M )

def error( ThetaT, X, Y):
    Error = h(ThetaT, X) - Y 
    return Error                    #Error is ( 1 , M ) 

def cost ( ThetaT, X, Y ):
    M = Y.shape[1]
    Factor = float(0.5 / M )    
    Error = error( ThetaT, X, Y) 
    Square_Error = (Error).dot(Error.T)
    Cost = Factor*Square_Error[0][0]   
    return Cost

def descent( ThetaT, X, Y ):
    M = Y.shape[1] 
    Factor = float(1/M)
    Error = error( ThetaT, X, Y) 
    Descent = Factor * Error.dot(X.T)    #Descent is ( 1 , N+1 )
    return Descent


LEARNING_RATES = [0.05]
ITERATIONS = 3000
DATASET_SIZE = [10] 
Training_Fraction = 0.8 
MAX_Degree = 9 
DEGREE = [ i for i in range(1 , MAX_Degree+1) ]

sys.stdout = open("Results.txt", "w")
os.makedirs("Hypothesis Functions", exist_ok=True)
os.makedirs("Cost vs Iteration", exist_ok=True)                        
for ALPHA in LEARNING_RATES : 

    Train_Costs =[]
    Test_Costs = []
    Final_ThetaT = []

    for M in DATASET_SIZE: 

        Dataset,X,Y = GenerateDataset(M) 
        Train_X, Train_Y, Test_X, Test_Y = Split( Dataset , Training_Fraction )

        Training_Cost = []
        Testing_Cost = []
        Learned_ThetaT = []

        for N in DEGREE :
            
            ThetaT = (0.00001)*np.random.randn( 1 , N+1 )
            Training_X = designMatrix( Train_X, N )
            Testing_X = designMatrix( Test_X, N ) 
             
            CostVec = []
            for i in range(0 , ITERATIONS): 
                CostVec.append( cost(ThetaT, Training_X, Train_Y) )
                ThetaT = ThetaT - ALPHA*descent(ThetaT, Training_X, Train_Y)

            Training_Cost.append( cost(ThetaT, Training_X, Train_Y) )
            Testing_Cost.append( cost(ThetaT, Testing_X, Test_Y))
            Learned_ThetaT.append( ThetaT[0] )  

            #2a)
            os.chdir("Hypothesis Functions")               
            plt.figure(figsize=(12,8))
            plt.plot( Train_X.T.tolist() , Train_Y.T.tolist() , 'g+' , markersize = 8 , label = 'Training Data' )
            plt.plot( Test_X.T.tolist() , Test_Y.T.tolist() , 'rx' , markersize = 7 , label = 'Testing Data' )    
            plt.plot( np.sort(X).T.tolist() , h(ThetaT , designMatrix(np.sort(X),N) ).T.tolist() , 'b.-' , label = 'Predicted Value')
            plt.xlabel("X Values")
            plt.ylabel("Y Values")
            plt.title("Hypothesis Function of Degree" + str(N) + " (M=" + str(M) + ")" )
            plt.legend()
            plt.grid(True)
            plt.savefig("Degree" + str(N) + "Curve_M_" + str(M) )
            #plt.show()
            plt.close()
            os.chdir("..")

            os.chdir("Cost vs Iteration")               
            plt.figure(figsize=(12, 8)) 
            plt.plot( np.resize( np.array([i+1 for i in range(0,ITERATIONS)]) , (ITERATIONS,1) ).tolist() , np.resize(np.array(CostVec), (len(CostVec),1) ) , 'b.-' )
            plt.xlabel("Iterations")
            plt.xscale("log")
            plt.ylabel("Cost")
            plt.title("Variation of Cost with Iteration for Degree" + str(N) + " (M=" + str(M) + ", Alpha=" + str(ALPHA) + ")" )            
            plt.grid(True)
            plt.savefig( ("Degree" + str(N) + "Curve_M_" + str(M) + "_Alpha_" + str((int)(ALPHA*1000)) + 'm'  ))
            #plt.show()
            plt.close() 
            os.chdir("..")

        Train_Costs.append(Training_Cost)
        Test_Costs.append(Testing_Cost)
        Final_ThetaT.append(Learned_ThetaT)

        #2b)

        plt.figure(figsize=(12,8))
        plt.plot( DEGREE , Training_Cost , 'bo' , label= 'Training Error' )
        plt.plot( DEGREE , Testing_Cost , 'go' ,  label= 'Testing Error' )
        plt.xlabel("Degree of Learning Curve")
        plt.ylabel("Mean Square Error") 
        plt.title("VARIATION of Errors with Degree of Learning Curve (M=" + str(M) + ")" )
        plt.legend()
        plt.grid(True)       
        plt.savefig( "Errors_vs_Features_M_" + str(M) )
        #plt.show()
        plt.close() 
        
        Columns = [ "Theta"+str(c) for c in range(0,MAX_Degree+1) ]
        Index = [ "Deg"+str(d) for d in DEGREE]
        DF = pd.DataFrame( Final_ThetaT[-1] , index = Index , columns = Columns )
        print("")
        print(("PARAMETERS LEARNED for M = " + str(M)).center(104) )
        print("-"*104)
        print(DF)   

        frame = { "Features": [ d+1 for d in DEGREE ] , "Training Error": Train_Costs[-1] , "Testing Error": Test_Costs[-1] , "Difference": [Test_Costs[-1][i]-Train_Costs[-1][i] for i in range(0, MAX_Degree)] }
        DF = pd.DataFrame(frame)
        DF.set_index("Features" , inplace=True )
        print("")  
        print(("VARIATION OF ERRORS WITH FEATURES, M="+str(M)).center(51) )
        print("-"*51 )
        print(DF)  

sys.stdout = standard_output
     