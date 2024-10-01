#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Driver file for linear n-mass oscillator
#
# Author:   Peter Manzl
# Date:     2024-04-08
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import exudyn as exu
from exudyn.utilities import *
from exudyn.processing import ParameterVariation

from simModels import SimulationModel, SliderCrank
from AISurrogateLib import * #MyNeuralNetwork, NeuralNetworkTrainingCenter, PVCreateData, NeuralNetworkStructureTypes, VariableShortForm, ExtendResultsFile, ParameterFunctionTraining
                   
from exudyn.plot import PlotSensor
import AISurrogateLib as aiLib

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
torch.set_num_threads(1)

useCUDA = torch.cuda.is_available()
useCUDA = False # CUDA support helps for fully connected networks > 256
# note that the current parameter variation, which is used to train 
# several networks in parallel does not support spawning threads with cuda. 

useRandomExcitation = True
useHarmonicExcitation = False

if useHarmonicExcitation == True and useRandomExcitation == True: 
    raise ValueError('useHarmonicExcitation and useRandomExcitation can not be both True!')
    
if __name__ == '__main__': #include this to enable parallel processing
    print('pytorch cuda=',useCUDA)

# currently 12 possible activation types
def GetActivationType(iAct): 
    activationFuncions = [nn.ReLU, nn.ELU, nn.GELU, nn.GLU, nn.Tanh, nn.CELU, nn.CELU,
                          nn.Hardsigmoid, nn.Hardtanh, nn.SiLU, nn.Mish, nn.LeakyReLU]
    return activationFuncions[iAct]

class CustomNNNonlinearOscillator(MyNeuralNetwork):
    def __init__(self, inputOutputSize = None, hiddenLayerSize = None, hiddenLayerStructure = ['L'], 
                 neuralNetworkTypeName = 'RNN', computeDevice = 'cpu', 
                 rnnNumberOfLayers = 1, resLayerList = [], hiddenLayerSizeList = [],
                 rnnNonlinearity = 'tanh', activationType = 0, typeNumber = 0, #'relu', None; 'tanh' is the default
                 ):
        super().__init__(inputOutputSize = inputOutputSize, hiddenLayerSize = hiddenLayerSize, hiddenLayerStructure = hiddenLayerStructure, 
                      neuralNetworkTypeName = neuralNetworkTypeName, computeDevice = computeDevice, 
                      rnnNumberOfLayers = rnnNumberOfLayers,rnnNonlinearity = rnnNonlinearity)
        self.myNN = self.customNetworks(inputOutputSize, hiddenLayerSize, typeNumber = typeNumber, resLayerList= resLayerList, activationType=activationType)
    
    def customNetworks(self, inputOutputSize, hiddenLayerSize, typeNumber = 0, activationType = 0, resLayerList = []): 
        activationType  = GetActivationType(activationType)
        print('initializing custom network for Slider-Crank; typeNumber: {}, activationType: {}'.format(typeNumber, activationType))    

        self.typeNumber  = typeNumber 
        self.resLayerList = resLayerList
        print('resLayerList: ', resLayerList)
        nnList = []        
        if typeNumber  == 0: 
            model = nn.Sequential(nn.Flatten(1))
            # model = nn.Sequential(nn.Linear(in_features, out_features))
            model.append(nn.Linear(np.prod(inputOutputSize[0]), hiddenLayerSize))
            model.append(activationType())
            model.append(nn.Linear(hiddenLayerSize, hiddenLayerSize))
            model.append(activationType())
            model.append(nn.Linear(hiddenLayerSize, inputOutputSize[1][0]))
            model.append(nn.Unflatten(-1, inputOutputSize[1]))
        elif typeNumber  == 1: 
            # model = nn.Sequential(nn.Flatten(1))
            model = nn.Sequential(nn.Conv1d(1, 1, 3, padding=1))
            # model.append(nn.Conv1d(1, 1, 3, 1))#  np.prod(inputOutputSize[0]), hiddenLayerSize))       
            model.append(activationType())
            model.append(nn.Linear(hiddenLayerSize-2, hiddenLayerSize))
            model.append(activationType())
            model.append(nn.Linear(hiddenLayerSize, inputOutputSize[1][0]))
            # model.append(nn.Unflatten(-1, inputOutputSize[1]))
            
            
        elif typeNumber  == 2: 
            # model = nn.Sequential(nn.Flatten(1))
            model = nn.Sequential(nn.Linear(np.prod(inputOutputSize[0]), hiddenLayerSize))
            model.append(activationType())
            # model.append(nn.Linear(hiddenLayerSize, hiddenLayerSize))
            model.append(nn.Conv1d(1,1,3,padding=1))
            # model.append(nn.Flatten(1))
            model.append(activationType())
            model.append(nn.Linear(hiddenLayerSize, inputOutputSize[1][0]))
            model.append(nn.Unflatten(-1, inputOutputSize[1]))
            
        return model 
    
    def forward(self, x): 
        nLayers = len(self.myNN)
            
        if self.typeNumber  == 0 or self.typeNumber == 1 or self.typeNumber == 2:
            if len(self.resLayerList) == 0: 
                for i in range(nLayers): 
                    x = self.myNN[i](x) 
            else: 
                xInput = torch.clone(x)
                if self.typeNumber == 1: 
                    x = x[:,:,2:] # only force
                # residual network: 
                for i in range(nLayers): 
                    x = self.myNN[i](x)
                    if i in self.resLayerList: 
                        if self.typeNumber == 0: 
                            x = x  + + self.myNN[0](xInput)  # flatten befor adding residual layer
                        else: 
                            x = x  + xInput # attention 
                        # print('add')
                    
                        
        if  self.typeNumber == 1 or self.typeNumber == 2 : 
            return x.view(-1, nnModel.myNN[-1].out_features, 1)
        return x

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# good results:

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#put this part out of if __name__ ... if you like to use multiprocessing\

parameterVariation = True

endTime=1
nMasses=1
useInitialValues=(nMasses==1)*0
useVelocities=(nMasses==1)*0
useFriction=False
nStepsTotal= 64 #63
variationMKD= False 

nnType = 'FFN' #'GRU'  # 0 RNN, 1 FFN


simModel = NonlinearOscillator(nStepsTotal=nStepsTotal, variationMKD=variationMKD,  useInitialValues=True, # useInitialVelocities=True, useInitialAngles=True, 
                         useRandomExcitation=useRandomExcitation, useHarmonicExcitation=useHarmonicExcitation, 
                         useVelocities=False, flagNoForce = False
                         )
simModel.CreateModel()
getDamping(simModel.mbs, 0.01)

nnModel = MyNeuralNetwork(inputOutputSize = simModel.GetInputOutputSizeNN(), # input and output size, 
                           neuralNetworkTypeName = 'FFN',
                           hiddenLayerSize = 40,
                           hiddenLayerStructure = ['L', 'R', 'L'],
                           computeDevice='cpu',
                           # typeNumber = 1, 
                           # activationType = 1
                           )

#MyNeuralNetwork()
nntc = NeuralNetworkTrainingCenter(nnModel, simulationModel=simModel, computeDevice='cuda' if useCUDA else 'cpu',
                                   verboseMode=0)
aiLib.moduleNntc = nntc #this informs the module about the NNTC (required for multiprocessing)

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#in this section we create or load data
nTraining=2048
nTest=512


    
dataFile = 'data/LinearOscillatorT'  +str(nTraining)+'-'+str(nTest)+'s'+\
            str(nStepsTotal) + 'randomExcitation' * bool(useRandomExcitation) \
                + 'HarmonicExcitation'*bool(useHarmonicExcitation) +'mkd' + str(int(variationMKD))\
                    +'vel'*bool(simModel.useVelocities) + '_noForce' * bool(simModel.flagNoForce) + '_test'
                # str(simModel.initVelRange[1]) +'t'+str(endTime)

createData = aiLib.checkCreateData(dataFile)
# createData = True  # can be overwritten to generate new training and test data sets. 

if __name__ == '__main__': #include this to enable parallel processing
    if createData:
        nntc.verboseMode = 1
        nntc.CreateTrainingAndTestData(nTraining=nTraining, nTest=nTest,
                                        parameterFunction=PVCreateData, #for multiprocessing
                                       # showTests=[0,1], #run SolutionViewer for this test
                                       )
        nntc.SaveTrainingAndTestsData(dataFile)
        # sys.exit()

if not createData and not parameterVariation:
    nntc.LoadTrainingAndTestsData(dataFile, nTraining=512, nTest=64)



#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__': #include this to enable parallel processing

    identifierString = 'E'
    storeModelName = 'model/'+simModel.GetModelName()+identifierString
    simModel.modelName = 'linear Oscillator'
    simModel.modelNameShort = 'linearOscillator'
    resultsFile = 'solution/res_'+simModel.GetModelNameShort()+'Res'+identifierString
    
    # in the  hiddenLayerStructureTyper variable some variants are shown
    #%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if parameterVariation:
        functionData = {'maxEpochs': 1500, # 10_000,
                        'nTraining':2048, # must not be more than previously created
                        'nTest':512,
                        'lossThreshold':1e-14, # threshold for early stopping. 
                        'lossLogInterval':10,
                        'learningRate': 2e-3,
                        'testEvaluationInterval':25, # or 100?
                        'neuralNetworkType':1, #0=RNN, 1=FFN, 2=CNN; currently only FFN used (and fully implemented)
                        'hiddenLayerSize':(nStepsTotal+2)*4,
                        'batchSize':128*4,
                        'storeModelName':storeModelName,
                        'modelName':simModel.GetModelName(),
                        'dataFile':dataFile,
                        'inputOutputSizeNN': simModel.GetInputOutputSizeNN(),  # for neural network input/outputs
                        'computeDevice': 'cuda' if useCUDA else 'cpu', 
                        'verboseMode': 1, 
                        'useBias': False, 
                        # 'customNetwork': CustomNNNonlinearOscillator, # 
                        } #additional parameters
        
        
        parameters = { 'hiddenLayerStructureType': [0, 29], 
                       "case": [0,1,2,3,5], 
                       }
        tStart = time.time()
        numRuns = 1
        for key, val in parameters.items():
            numRuns *= len(val) 
        [parameterDict, valueList] = ParameterVariation(parameterFunction=ParameterFunctionTraining, 
                                                      parameters=parameters,
                                                        useMultiProcessing=(numRuns>1),
                                                      # numberOfThreads=4, # explicitly set number of parallel runs. 
                                                     resultsFile=resultsFile+'.txt', 
                                                     addComputationIndex=True,
                                                     parameterFunctionData=functionData)
        values, nnModels = [], []
        for val in valueList: 
            values += [val[0]]
            nnModels += [val [1]]
        CPUtime=time.time()-tStart
        print('training variation took:',round(CPUtime,2),'s')
        functionData['CPUtime']=CPUtime
        
        #++++++++++++++++++++++++++++++++++++++
        #store data in readable format
        ExtendResultsFile(resultsFile+'.txt', functionData, parameterDict)
        
        dataDict = {'version':1}
        dataDict['parameters'] = parameters
        dataDict['parameterDict'] = parameterDict
        dataDict['values'] = values
        dataDict['functionData'] = functionData

        with open(resultsFile+'.npy', 'wb') as f:
            np.save(f, dataDict, allow_pickle=True) #allow_pickle=True for lists or dictionaries

        #%%++++++++++++++++++++++++++++++++++++++
        #show loss, test error over time, ...
        resultsFileNpy = resultsFile+'.npy'
        nntc.PlotTrainingResults(resultsFileNpy, dataFile)
        
        
        #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #check solution of one of the neural networks
        nntc.LoadNNModel(storeModelName+str(0)+'.pth') #best model: #5, 2500 epochs, lr0.001, 'L', NT512, HL128
        error = nntc.EvaluateModel(plotTests=[0,1,2,3,4], plotTrainings=[0,1,2,3,4], plotVars=['time','ODE2'], computeRMSE =True)


        

    
#%% 
if True: 
    #%% visualization of matrices for linear layers
    netTest = valueList[0][1]
    w1, w2 = netTest.myNN[1].weight.detach().numpy(), netTest.myNN[2].weight.detach().numpy() # np.eye(64)# , netTest.myNN[2].weight.detach().numpy()
    w3 = netTest.myNN[3].weight.detach().numpy()
    flagNoBias = False
    if not(netTest.myNN[1].bias is None): 
        flagNoBias = True
        b1, b2 = netTest.myNN[1].bias.detach().numpy(), netTest.myNN[2].bias.detach().numpy()
        A1 = np.concatenate((w1, np.array([b1]).T), 1)
        A2 = np.concatenate((w2, np.array([b2]).T), 1)
    else: 
        A1, A2 = w1, w2
    if 0: 
        plt.figure()
        plt.title('A1')
        plt.imshow(A1, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.figure()
        plt.title('A2')
        plt.imshow(A2, cmap='hot', interpolation='nearest')
        plt.colorbar()
    

    w_total = w3 @ w2 @ w1    
    if flagNoBias: 
        b_total = w2 @ b1 + b2
        A_total = np.concatenate((w_total, np.array([b_total]).T), 1)
    else: 
        A_total = w_total
    plt.figure()
    # plt.title('weights/biases after mult')
    plt.imshow((A_total[:,:]) , cmap='viridis', interpolation='nearest')
    # plt.imshow((A_total) , cmap='PiYG', interpolation='nearest')
    
    plt.colorbar()
#%% 
    if 0: 
    #%% 
        plt.figure()
        plt.plot(A_total[:,0], label='column 0: initial position')    
        # plt.plot(A_total[:,10], label='')

        plt.grid()
        # plt.xlabel('output index j\n t(j) = hj)
        plt.xlabel('output index j\n ' + r't(j) = $h\,j$')
        plt.ylabel(r'$x_{out}$')
        plt.legend()
        plt.savefig('LinearSystem_c0.png')
        plt.plot(A_total[:,25], label='column 25: force at j=23')
        plt.legend()
        plt.savefig('LinearSystem_c1.png')
        
    #%%
    if 0: 
        for i in range(10): 
            plt.gcf().get_children()[-1].remove()
        plt.legend(['1 hidden layer'] + ['__no__legend__']*4 + ['shallow network'])
        