
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
import copy
import random



def compileModel(mList,numModel):
    for counterCompile in range(numModel):
        mList[counterCompile].compile(optimizer='Nadam', loss='binary_crossentropy')
        

def getWeightsBias(mList,weightsList,biasList,numModel):
    for counterLayer in range(3):
        for counterModel in range(numModel):
            weightsList[counterLayer][counterModel]=mList[counterModel].layers[counterLayer].get_weights()[0]
            biasList[counterLayer][counterModel]=mList[counterModel].layers[counterLayer].get_weights()[1]
    return weightsList,biasList

def buildModelList(numUnitLayer0,numUnitLayer1,numUnitLayer2,xTrain,numModel):

    nn=Sequential([
            Dense(numUnitLayer0, activation='relu', input_dim=xTrain.shape[1], kernel_regularizer=regularizers.l2(0.01)),
            Dense(numUnitLayer1, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dense(numUnitLayer2, activation='sigmoid')
        ])
    return nn

def bestWeightsBias(halfModel,xTrain,numUnitLayer,indexBest,weightsList,biasList):
    bestWeightsLayer0=np.ones((halfModel,xTrain.shape[1],numUnitLayer[0]))
    bestWeightsLayer1=np.ones((halfModel,numUnitLayer[0],numUnitLayer[1]))
    bestWeightsLayer2=np.ones((halfModel,numUnitLayer[1],numUnitLayer[2]))
    bestBiasLayer0=np.ones((halfModel,numUnitLayer[0]))
    bestBiasLayer1=np.ones((halfModel,numUnitLayer[1]))
    bestBiasLayer2=np.ones((halfModel,numUnitLayer[2]))
    bestWeightsList=[bestWeightsLayer0,bestWeightsLayer1,bestWeightsLayer2]
    bestBiasList=[bestBiasLayer0,bestBiasLayer1,bestBiasLayer2]
    
    for counterWeightsBias in range(halfModel):
        b=int(indexBest[counterWeightsBias])
        bestWeightsList[0][counterWeightsBias]=weightsList[0][b]
        bestWeightsList[1][counterWeightsBias]=weightsList[1][b]
        bestWeightsList[2][counterWeightsBias]=weightsList[2][b]
        bestBiasList[0][counterWeightsBias]=biasList[0][b]
        bestBiasList[1][counterWeightsBias]=biasList[1][b]
        bestBiasList[2][counterWeightsBias]=biasList[2][b]
    return bestWeightsList,bestBiasList
               
        
        
def aucrocScore(prediction,numModel,yTest):
    aucrocScore=np.ones((numModel))
    for aucrocCounter in range(numModel):
        aucrocScore[aucrocCounter]=roc_auc_score(yTest, prediction[aucrocCounter])
    return aucrocScore

def selectBest(aucrocScore,numModel):
    indexBest=np.ones(((int(numModel/2))))
    for selectCounter in range(int(numModel/2)):
        a=np.argmax(aucrocScore)
        indexBest[selectCounter]=a #using a to avoid index error
        aucrocScore[a]=-1
    return indexBest

def mutation1(bestWeightsList,bestBiasList,mutationRange,mutSelectProb,numLayer):
    '''In this function, I mutate the bestWeightsList and bestBiasList'''
    '''Using normal distribution to construct an mutation array with the same size as bestWeightsList and bestBiasList'''
    '''Then we add the two array together and get the muteted bestWeightsList and bestBiasList'''
    for counterMut in range(numLayer):
        '''Mutation of Weights'''
#        mutWeightsMatLayer=np.random.uniform(-mutationRange,mutationRange,
#                                             bestWeightsList[counterMut].shape)
        mutWeightsMatLayer=np.random.normal(0,1,bestWeightsList[counterMut].shape)
#        mutWeightsMatSelectLayer=np.random.choice([1, 0], size=bestWeightsList[counterMut].shape, 
#                                               p=[mutSelectProb,1-mutSelectProb])
#        mutWeightsMatLayer=mutWeightsMatLayer*mutWeightsMatSelectLayer
        bestWeightsList[counterMut]=bestWeightsList[counterMut]+mutWeightsMatLayer
        '''Mutation of Bias'''
#        mutBiasMatLayer=np.random.uniform(-mutationRange,mutationRange,bestBiasList[counterMut].shape)
#        mutBiasMatSelectLayer=np.random.choice([1, 0], size=bestBiasList[counterMut].shape, 
#                                               p=[mutSelectProb,1-mutSelectProb])
#        mutBiasMatLayer=mutBiasMatLayer*mutBiasMatSelectLayer
        mutBiasMatLayer=np.random.normal(0,1,bestBiasList[counterMut].shape)
        bestBiasList[counterMut]=bestBiasList[counterMut]+mutBiasMatLayer
        return bestWeightsList, bestBiasList
    
def globalHybrid(numLocal,globalWeightsList,globalBiasList,numFeature,numUnitLayer,globalHybridPercent):
    counter=0
    numCombination=int(((numLocal*(numLocal+1)))/2)
    combinationWeightsLayer0=np.ones((numCombination,numFeature,numUnitLayer[0]))
    combinationWeightsLayer1=np.ones((numCombination,numUnitLayer[0],numUnitLayer[1]))
    combinationWeightsLayer2=np.ones((numCombination,numUnitLayer[1],numUnitLayer[2]))
    combinationBiasLayer0=np.ones((numCombination,numUnitLayer[0]))
    combinationBiasLayer1=np.ones((numCombination,numUnitLayer[1]))
    combinationBiasLayer2=np.ones((numCombination,numUnitLayer[2]))
    for counterCombination in range(numLocal-1):
        for counterAdd in range(1,numLocal-counterCombination):
            a=np.ones((int(globalHybridPercent*(globalWeightsList[0][0].size))))
            b=np.zeros((globalWeightsList[0][0].size-int(globalHybridPercent*(globalWeightsList[0][0].size))))
            a=np.append(a,b)
            np.random.shuffle(a)
            a=a.reshape((globalWeightsList[0][0].shape))
            c=(-(a-1))
            combinationWeightsLayer0[counter]=a*globalWeightsList[counterCombination][0]+c*globalWeightsList[counterCombination+counterAdd][0]
            
            a=np.ones((int(globalHybridPercent*(globalBiasList[0][0].size))))
            b=np.zeros((globalBiasList[0][0].size-int(globalHybridPercent*(globalBiasList[0][0].size))))
            a=np.append(a,b)
            np.random.shuffle(a)
            a=a.reshape((globalBiasList[0][0].shape))
            c=(-(a-1))
            combinationBiasLayer0[counter]=a*globalBiasList[counterCombination][0]+c*globalBiasList[counterCombination+counterAdd][0]
            
            a=np.ones((int(globalHybridPercent*(globalWeightsList[0][1].size))))
            b=np.zeros((globalWeightsList[0][1].size-int(globalHybridPercent*(globalWeightsList[0][1].size))))
            a=np.append(a,b)
            np.random.shuffle(a)
            a=a.reshape((globalWeightsList[0][1].shape))
            c=(-(a-1))
            combinationWeightsLayer1[counter]=a*globalWeightsList[counterCombination][1]+c*globalWeightsList[counterCombination+counterAdd][1]
            
            
            a=np.ones((int(globalHybridPercent*(globalBiasList[0][1].size))))
            b=np.zeros((globalBiasList[0][1].size-int(globalHybridPercent*(globalBiasList[0][1].size))))
            a=np.append(a,b)
            np.random.shuffle(a)
            a=a.reshape((globalBiasList[0][1].shape))
            c=(-(a-1))
            combinationBiasLayer1[counter]=a*globalBiasList[counterCombination][1]+c*globalBiasList[counterCombination+counterAdd][1]
        
            a=np.ones((int(globalHybridPercent*(globalWeightsList[0][2].size))))
            b=np.zeros((globalWeightsList[0][2].size-int(globalHybridPercent*(globalWeightsList[0][2].size))))
            a=np.append(a,b)
            np.random.shuffle(a)
            a=a.reshape((globalWeightsList[0][2].shape))
            c=(-(a-1))
            combinationWeightsLayer2[counter]=a*globalWeightsList[counterCombination][2]+c*globalWeightsList[counterCombination+counterAdd][2]
   
            a=np.ones((int(globalHybridPercent*(globalBiasList[0][2].size))))
            b=np.zeros((globalBiasList[0][2].size-int(globalHybridPercent*(globalBiasList[0][2].size))))
            a=np.append(a,b)
            np.random.shuffle(a)
            a=a.reshape((globalBiasList[0][2].shape))
            c=(-(a-1))
            combinationBiasLayer2[counter]=a*globalBiasList[counterCombination][2]+c*globalBiasList[counterCombination+counterAdd][2]        
            
            counter=counter+1
            
    combinationWeightsList=[combinationWeightsLayer0,combinationWeightsLayer1,combinationWeightsLayer2]
    combinationBiasList=[combinationBiasLayer0,combinationBiasLayer1,combinationBiasLayer2]
    return combinationWeightsList,combinationBiasList


def initWeightsBiasList0(numModel,xTrain,numUnitLayer):
    
    weightsLayer0=np.random.uniform(-0.21,0.21,size=(numModel,xTrain.shape[1],numUnitLayer[0]))
    weightsLayer1=np.random.uniform(-0.35,0.62,size=(numModel,numUnitLayer[0],numUnitLayer[1]))
    weightsLayer2=np.random.uniform(-2.24,1.31,size=(numModel,numUnitLayer[1],numUnitLayer[2]))
    biasLayer0=np.random.uniform(0,0.38,size=(numModel,numUnitLayer[0]))
    biasLayer1=np.random.uniform(-0.03,0.75,size=(numModel,numUnitLayer[1]))
    biasLayer2=np.random.uniform(0.25,0.45,size=(numModel,numUnitLayer[2]))
    return [weightsLayer0,weightsLayer1,weightsLayer2],[biasLayer0,biasLayer1,biasLayer2]

def initWeightsBiasList1(weightsListCopy,biasListCopy):
    weightsLayer0=copy.deepcopy(weightsListCopy[0])
    weightsLayer1=copy.deepcopy(weightsListCopy[1])
    weightsLayer2=copy.deepcopy(weightsListCopy[2])
    biasLayer0=copy.deepcopy(biasListCopy[0])
    biasLayer1=copy.deepcopy(biasListCopy[1])
    biasLayer2=copy.deepcopy(biasListCopy[2])
    return [weightsLayer0,weightsLayer1,weightsLayer2],[biasLayer0,biasLayer1,biasLayer2]

def loadWeightsBias(weightsList,biasList,numModel,mList):
    for counterGetWeigth in range(numModel):
        wbLayer0=[weightsList[0][counterGetWeigth],biasList[0][counterGetWeigth]]
        mList[counterGetWeigth].layers[0].set_weights(wbLayer0)
        
        wbLayer1=[weightsList[1][counterGetWeigth],biasList[1][counterGetWeigth]]
        mList[counterGetWeigth].layers[1].set_weights(wbLayer1)
    
        wbLayer2=[weightsList[2][counterGetWeigth],biasList[2][counterGetWeigth]]
        mList[counterGetWeigth].layers[2].set_weights(wbLayer2)
        
def loadWeightsBias2(weightsList,biasList,numModel,mList):
    for counterGetWeigth in range(numModel):
        wbLayer0=[weightsList[0][counterGetWeigth],biasList[0][counterGetWeigth]]
        mList[counterGetWeigth].layers[0].set_weights(wbLayer0)
        
        wbLayer1=[weightsList[1][counterGetWeigth],biasList[1][counterGetWeigth]]
        mList[counterGetWeigth].layers[1].set_weights(wbLayer1)
    
        wbLayer2=[weightsList[2][counterGetWeigth],biasList[2][counterGetWeigth]]
        mList[counterGetWeigth].layers[2].set_weights(wbLayer2)


def selectCombination(numLocal, numModel, combinationWeightsList,combinationBiasList,xLocalTrain,numUnitLayer):
    weightsLayer0=np.ones((numModel,xLocalTrain.shape[1],numUnitLayer[0]))
    weightsLayer1=np.ones((numModel,numUnitLayer[0],numUnitLayer[1]))
    weightsLayer2=np.ones((numModel,numUnitLayer[1],numUnitLayer[2]))
    biasLayer0=np.ones((numModel,numUnitLayer[0]))
    biasLayer1=np.ones((numModel,numUnitLayer[1]))
    biasLayer2=np.ones((numModel,numUnitLayer[2]))
    numCombination=int(((numLocal*(numLocal+1)))/2)
    rangeSample=random.sample(range(numCombination), numModel)
    counterModel=0
    for counterSample in rangeSample:
        weightsLayer0[counterModel]=weightsLayer0[counterModel]*combinationWeightsList[0][counterSample]
        weightsLayer1[counterModel]=weightsLayer1[counterModel]*combinationWeightsList[1][counterSample]
        weightsLayer2[counterModel]=weightsLayer2[counterModel]*combinationWeightsList[2][counterSample]     
        biasLayer0[counterModel]=biasLayer0[counterModel]*combinationBiasList[0][counterSample]
        biasLayer1[counterModel]=biasLayer1[counterModel]*combinationBiasList[1][counterSample]
        biasLayer2[counterModel]=biasLayer2[counterModel]*combinationBiasList[2][counterSample]
        counterModel=counterModel+1
    return [weightsLayer0,weightsLayer1,weightsLayer2],[biasLayer0,biasLayer1,biasLayer2]
    
'''this function using layers initial weights and bias to set weights and bias'''     
def iniWeightsBiasList2(numUnitLayer,numLayer,numModel,X):
    K.clear_session()
    nnInitial=Sequential([
        Dense(numUnitLayer[0], activation='relu', input_dim=X.shape[1], kernel_regularizer=regularizers.l2(0.01)),
        Dense(numUnitLayer[1], activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dense(numUnitLayer[2], activation='sigmoid')
    ])
    weightsLayer0=np.ones((numModel,X.shape[1],numUnitLayer[0]))
    weightsLayer1=np.ones((numModel,numUnitLayer[0],numUnitLayer[1]))
    weightsLayer2=np.ones((numModel,numUnitLayer[1],numUnitLayer[2]))
    biasLayer0=np.ones((numModel,numUnitLayer[0]))
    biasLayer1=np.ones((numModel,numUnitLayer[1]))
    biasLayer2=np.ones((numModel,numUnitLayer[2]))
    weightsList,biasList=[weightsLayer0,weightsLayer1,weightsLayer2],[biasLayer0,biasLayer1,biasLayer2]
    for counterLayer in range(numLayer):
        for counterModel in range(numModel):
            weightsList[counterLayer][counterModel]=nnInitial.layers[counterLayer].get_weights()[0]
            biasList[counterLayer][counterModel]=nnInitial.layers[counterLayer].get_weights()[1]
    return weightsList, biasList
            
    

    
def shuffleData(x,y):
    '''remove patient names from the sheet, for later processing'''
    X=x.values[0:x.shape[0],1:x.shape[1]]
    Xcopy=copy.deepcopy(X)
    Y=y.values[0:x.shape[0],1]
    Ycopy=copy.deepcopy(Y)
    
    '''shuffle the data'''
    dataIndex=random.sample(range(X.shape[0]),X.shape[0])
    for counterDataShuffle in range(X.shape[0]):
        X[counterDataShuffle]=Xcopy[dataIndex[counterDataShuffle]]
        Y[counterDataShuffle]=Ycopy[dataIndex[counterDataShuffle]]
    print('data shuffle done')
    return X,Y

def initParameter(numModel,X,numUnitLayer,numUnitLayer0,numUnitLayer1,numUnitLayer2,numLayer,numElement):
    '''Let the complier build a neural network model using default parameter'''
    '''Then copy those parameters and change into required dimension'''
    weightsGroupList=[]
    biasGroupList=[]
    nnInitParameter=buildModelList(numUnitLayer0,numUnitLayer1,numUnitLayer2,X,numModel)
    weightsLayer0=np.ones((numModel,X.shape[1],numUnitLayer[0]))
    weightsLayer1=np.ones((numModel,numUnitLayer[0],numUnitLayer[1]))
    weightsLayer2=np.ones((numModel,numUnitLayer[1],numUnitLayer[2]))
    biasLayer0=np.ones((numModel,numUnitLayer[0]))
    biasLayer1=np.ones((numModel,numUnitLayer[1]))
    biasLayer2=np.ones((numModel,numUnitLayer[2]))
    weightsInitList=[weightsLayer0,weightsLayer1,weightsLayer2]
    biasInitList=[biasLayer0,biasLayer1,biasLayer2]
    for counterLayer in range(numLayer):
        for counterModel in range(numModel):
            weightsInitList[counterLayer][counterModel]=nnInitParameter.layers[counterLayer].get_weights()[0]
            biasInitList[counterLayer][counterModel]=nnInitParameter.layers[counterLayer].get_weights()[1]
    for counterElement in range(numElement):
        weightsGroupList.append(weightsInitList)
        biasGroupList.append(biasInitList)
    return weightsGroupList, biasGroupList

def selectDataset(counterElement,X,Y,step):
    xLocal=X[counterElement*step:((counterElement+1)*step)]
    xLocalTrain=xLocal[0:int(step*0.8)]
    xLocalTest=xLocal[int(step*0.8):]
    yLocal=Y[counterElement*step:((counterElement+1)*step)]
    yLocalTrain=yLocal[0:int(step*0.8)]
    yLocalTest=yLocal[int(step*0.8):]
    return xLocalTrain,xLocalTest,yLocalTrain,yLocalTest

def averageParameter(sumDataPiece,numUnitLayer,X,numElement,numLayer,weightsGroupList,biasGroupList):
    '''Average parameters based on the percentage of data pieces used for training'''
    '''and the final result is parameters of just one neural net'''
    percent=sumDataPiece/sumDataPiece.sum()
    finalWeightsLayer0=np.zeros((X.shape[1],numUnitLayer[0]))
    finalWeightsLayer1=np.zeros((numUnitLayer[0],numUnitLayer[1]))
    finalWeightsLayer2=np.zeros((numUnitLayer[1],numUnitLayer[2]))
    finalWeightsList=[finalWeightsLayer0,finalWeightsLayer1,finalWeightsLayer2]
    finalBiasLayer0=np.zeros((numUnitLayer[0]))
    finalBiasLayer1=np.zeros((numUnitLayer[1]))
    finalBiasLayer2=np.zeros((numUnitLayer[2]))
    finalBiasList=[finalBiasLayer0,finalBiasLayer1,finalBiasLayer2]
    
    for counterList in range(numElement):
        for counterLayer in range(numLayer):
            finalWeightsList[counterLayer]=finalWeightsList[counterLayer]+percent[counterList]*weightsGroupList[counterList][counterLayer][0]
            finalBiasList[counterLayer]=finalBiasList[counterLayer]+percent[counterList]*biasGroupList[counterList][counterLayer][0]
    return finalWeightsList, finalBiasList
       
        
