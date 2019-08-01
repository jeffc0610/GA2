
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.models import load_model
from keras import models
from keras import layers
from keras.models import Model
from keras import optimizers
from keras.layers import Dense, Activation
from keras.layers import Input, Dense, Dropout
from keras.models import Model, load_model
from keras import regularizers
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score
import NN_GA as ga
import copy
from sklearn.model_selection import train_test_split
import random
#--------load data----------
x=pd.read_csv(r"C:\document\python\python\predict mortality\data\drug_table_mortality_with_no.csv")
y=pd.read_csv(r"C:\document\python\python\predict mortality\data\mortality_table.csv")

'''shuffle the dataset by changing the order between rows and rows randomly, 
each row represent one patient's individual informaiton'''
X,Y=ga.shuffleData(x,y)


'''the number of node in each layer in neural network '''
numUnitLayer0=4
numUnitLayer1=2
numUnitLayer2=1
numLayer=3
numUnitLayer=[numUnitLayer0,numUnitLayer1,numUnitLayer2]#improve convenience for later iteration



'''The number of model in each node'''
numModel=10####################################(The number of model in each node)changable
halfModel=int(numModel/2)
'''The number of mutation for training one time'''
numLoop=70##############(The number of mutation in one training process)changable
mutationRange=0.2#(Not used in this version)
mutSelectProb=0.5#（Not used in this version）
numFeature=X.shape[1]#The total number of madicine kind


numLocal=8##(Representing how many datasets are splitted into, which is the number of hospital in reality)changable
numElement=int(numLocal/2)#The number of subdataset used in one second loop
numSelectBestGlobal=2#(Not used in this version)
step=3000##(the number of data pieces in each sub dataset)changable
globalHybridPercent=0.5#NOT used in this version
bigLoop=4######(The number of first level loop executed)####changable

numSelectDataset=4##(The number of second level loop executed)####changeable
'''initial testset'''
xFinalTest=X[25000:]#####################################################
yFinalTest=Y[25000:]#########################################################

finalScore=[]#restore the aucroc score for every averaged loop



'''build an array to restore the number of data '''
sumDataPiece=np.zeros((numElement))

'''initial parameters'''
'''every model have the same parameter at the beginning'''
weightsGroupList, biasGroupList=ga.initParameter(numModel,X,
                                                   numUnitLayer,
                                                   numUnitLayer0,numUnitLayer1,numUnitLayer2,
                                                   numLayer,numElement)


for counterBigLoop in range(bigLoop):
    print('Number of bigloop:',bigLoop)
    '''Return the averaged parameters back to the beginning of the second level loop'''
    '''Use the averaged parameters to construct 'weightsGlobalList'and'biasGlobalList'''
    if counterBigLoop!=0:
        for counterElement in range(numElement):           
            for counterLayer in range(numLayer):
                for counterModel in range(numModel):
                    weightsGroupList[counterElement][counterLayer][counterModel]=finalWeightsList[counterLayer]
                    biasGroupList[counterElement][counterLayer][counterModel]=finalBiasList[counterLayer]
       
    for counterSelectDataset in range(numSelectDataset):
        '''This is the second level loop'''
        '''This loop will allow every model to combine different dataset for numSelectDataset times'''
        '''In this level loop, we stimulate the process that the distributor distribute parameter models to
        local hostital'''
        '''But the code is written based on models perspect, randomly select a dataset(a local hospital's information)
        to combine with a model choice orderly'''
        '''In this code, the data onwer would know the number order of the model parameters, but there
        is no certain links between dataset and local hostipal, so that the analyzer won't figure out how 
        models' parameters is trained, and which local hostital's data is used, just be informed the 
        medels' parameter and the number of data pieces used to train a certain individual model parameter'''
        
        print('Number of select dataset:', counterSelectDataset)
        K.clear_session()
        '''Create a random array of index for dataset selection'''
        datasetIndex=random.sample(range(numLocal),numElement)
        for counterDatasetIndex in range(numElement):
            print(counterDatasetIndex)
            '''To randomly select a dataset for training'''
            counterElement=datasetIndex[counterDatasetIndex]
            xLocalTrain,xLocalTest,yLocalTrain,yLocalTest=ga.selectDataset(datasetIndex[counterDatasetIndex],
                                                                           X,Y,step)
            '''Build up an array to store the number of data pieces used for 
            training each node(comprised of 10 neural networks)'''
            sumDataPiece[counterDatasetIndex]=sumDataPiece[counterDatasetIndex]+xLocalTrain.shape[0]+xLocalTest.shape[0]
    
            
            K.clear_session()
            mList=[]
            for counterList in range(numModel):
                mList.append(ga.buildModelList(numUnitLayer0,numUnitLayer1,numUnitLayer2,xLocalTrain,numModel))
            
            '''Select the node used for training'''
            weightsList=weightsGroupList[counterDatasetIndex]
            biasList=biasGroupList[counterDatasetIndex]
            '''Training'''
            '''Using genetic algorithm training neural network'''
            for counterLoop in range(numLoop):
                '''Set parameters for training model'''
                ga.loadWeightsBias(weightsList,biasList,numModel,mList)
                ga.compileModel(mList,numModel)
                '''Evaluate each model(a neural net) for later selection'''
                prediction=np.ones((numModel,xLocalTest.shape[0]))
                for counterPrediction in range(numModel):
                    prediction[counterPrediction]=mList[counterPrediction].predict(xLocalTest).reshape((1,xLocalTest.shape[0]))   
                aucrocScore=np.ones((numModel))
                aucrocScore=ga.aucrocScore(prediction,numModel,yLocalTest)
                '''Print current AUCROC score every 10 loop'''
                if counterLoop%10==0 or counterLoop==(numLoop-1):
                   print(aucrocScore.max())  
                '''Select the half models with better AUCROC score and store their index'''   
                indexBest=np.ones(((int(numModel/2)))) #select the index of the best scores
                indexBest=ga.selectBest(aucrocScore,numModel)
                '''Retrieve the parameters from training models'''
                weightsList,biasList=ga.getWeightsBias(mList,weightsList,biasList,numModel)#Maybe this statement is unnecessary
                '''Select model training parameters with better AUCROC score'''
                '''we build up two new list to store those parameters'''
                bestWeightsList,bestBiasList=ga.bestWeightsBias(halfModel,xLocalTrain,numUnitLayer,indexBest,weightsList,biasList)
                '''Store parameters with better AUCROC in the first half index of weightsList and biasList'''   
                for counterFeedBack in range(len(numUnitLayer)):
                    weightsList[counterFeedBack][:int(numModel/2)]=bestWeightsList[counterFeedBack]
                    biasList[counterFeedBack][:int(numModel/2)]=bestBiasList[counterFeedBack]
                '''mutation'''
                bestWeightsList,bestBiasList=ga.mutation1(bestWeightsList,bestBiasList,
                                                           mutationRange,mutSelectProb,numLayer)
                '''Store the mutated parameters in the second half index of weightsList and biasList'''
                for counterFeedBack in range(numLayer):
                    weightsList[counterFeedBack][int(numModel/2):]=bestWeightsList[counterFeedBack]
                    biasList[counterFeedBack][int(numModel/2):]=bestBiasList[counterFeedBack]  
            '''Training END'''
    
    '''Average parameters'''
    '''Using the best model in each code, which is sorted at the first position of each node'''
    finalWeightsList,finalBiasList=ga.averageParameter(sumDataPiece,numUnitLayer,X,
                                                       numElement,numLayer,weightsGroupList,biasGroupList)
    
    '''build a model to evaluate the final result after average'''
    finalModel=ga.buildModelList(numUnitLayer0,numUnitLayer1,numUnitLayer2,xFinalTest,numModel)
    
    for counterLayer in range(numLayer):
        wbLayer=[finalWeightsList[counterLayer],finalBiasList[counterLayer]]
        finalModel.layers[counterLayer].set_weights(wbLayer)
    
    prediction=finalModel.predict(xFinalTest)
    aucrocScore=roc_auc_score(yFinalTest, prediction)
    finalScore.append(aucrocScore)
    print(aucrocScore)
 
