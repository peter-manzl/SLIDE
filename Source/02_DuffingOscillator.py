#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Test model for linear n-mass oscillator
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
from exudyn.plot import PlotSensor


from simModels import SimulationModel, SliderCrank, NonlinearOscillatorContinuous
from AISurrogateLib import * #MyNeuralNetwork, NeuralNetworkTrainingCenter, PVCreateData, NeuralNetworkStructureTypes, VariableShortForm, ExtendResultsFile, ParameterFunctionTraining
import AISurrogateLib as aiLib
            

# from nnNonlinearOscillator import GetActivationType, CustomNNNonlinearOscillator
# from nnSliderCrank import CustomNNSliderCrank
import sys
import time
import numpy as np
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
useHarmonicExcitation = True

    
if __name__ == '__main__': #include this to enable parallel processing
    print('pytorch cuda=',useCUDA)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#put this part out of if __name__ ... if you like to use multiprocessing

parameterVariation = True
nMasses=1
useInitialValues=(nMasses==1)*0
useVelocities=(nMasses==1)*0
useFriction=True


variationMKD= False 
flagDuffing = True
useFriction = True
useErrorEstimator = True

tEnd = 3
nStepsTotal = 100
k, d, m = 1600, 8, 1
w0 = np.sqrt(k/m)
D = d/(2*m*w0)
# A = np.exp(-(w0*D*t)) # analytic solution from the linear system
tDamped = -np.log(0.004)/(w0*D)
nDamped = nStepsTotal*(tDamped/tEnd)
nOut  = int(nStepsTotal - np.ceil(nDamped)) # - 12


simModel = NonlinearOscillatorContinuous(nStepsTotal=nStepsTotal, useInitialValues=True, 
             frictionForce=0., nMasses=1, endTime=tEnd, scalBaseMotion = 0.2, nonlinearityFactor = 0.5, #factor for cubic nonlinearity
             initUfact = 1., initVfact = 10., nOutputSteps = nOut, useVelocities = False, 
             useHarmonicExcitation=useHarmonicExcitation, useRandomExcitation = useRandomExcitation)

simModel.CreateModel() # "assemble" model to be ready for simulation
getDamping(simModel.mbs, 0.01)

# the following models are only placeholders / dummys: 
nnModel = MyNeuralNetwork(inputOutputSize = simModel.GetInputOutputSizeNN(), # input and output size, 
                           neuralNetworkTypeName = 'FFN',
                           hiddenLayerSize = 40,
                           hiddenLayerStructure = ['L', 'R', 'L'],
                           computeDevice='cpu',
                           # typeNumber = 1, 
                           # activationType = 1
                           )

nnModelEst = MyNeuralNetwork(inputOutputSize = simModel.GetInputOutputSizeNN(), # input and output size, 
                           neuralNetworkTypeName = 'FFN',
                           hiddenLayerSize = 5,
                           hiddenLayerStructure = ['L', 'R', 'L'])

nntc = NeuralNetworkTrainingCenter(nnModel, simulationModel=simModel, computeDevice='cuda' if useCUDA else 'cpu',
                                   verboseMode=0, nnModelEst = nnModelEst)
aiLib.moduleNntc = nntc # this informs the module about the NNTC (multiprocessing)



#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#in this section we create or load data
nTraining= 4096
nTest= 1024


    
dataFile = 'data/NonlinearOscillatorDampedT'  +str(nTraining)+'-'+str(nTest)+'st'+\
            str(nStepsTotal) + '_'  + str(nOut) +'-randInp' * bool(useRandomExcitation) \
                + '-HarmonicInp'*bool(useHarmonicExcitation) +'-mkd' + str(int(variationMKD)) \
                + '-Duffing'* bool(flagDuffing) + str(simModel.nonlinearityFactor)*bool(flagDuffing) \
                + '-friction' * useFriction + '-velOut'*bool(simModel.useVelocities) \
                + 'inputScaling{}'.format(simModel.scalBaseMotion)
                # str(simModel.initVelRange[1]) +'t'+str(endTime)

createData = aiLib.checkCreateData(dataFile)
# createData = True   # can manually be overwritten to force generating new training and test data sets

if __name__ == '__main__': #include this to enable parallel processing
    if createData:
        nntc.verboseMode = 1
        nntc.CreateTrainingAndTestData(nTraining=nTraining, nTest=nTest,
                                        parameterFunction=PVCreateData, #for multiprocessing
                                       # showTests=[0,1], #run SolutionViewer for this test
                                       )
        nntc.SaveTrainingAndTestsData(dataFile)


if not createData and not parameterVariation:
    nntc.LoadTrainingAndTestsData(dataFile, nTraining=100, nTest=16) # for testint load a certain number of data


#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__': # needed to enable parallel processing of several trainings
    identifierString = 'G1' # change identifier string to not overwrite trained models / data
    storeModelName = 'model/'+simModel.GetModelName()+identifierString
    resultsFile = 'solution/res_'+simModel.GetModelNameShort()+'Res'+identifierString
    # %%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    functionData = {'maxEpochs':1200, # 10_000,
                    'nTraining':1024*4,
                    'nTest':64*2,
                    'lossThreshold':1e-10,
                    'lossLogInterval':20,
                    'testEvaluationInterval':25, # or 100?
                    'hiddenLayerSize': 100*2, 
                    'neuralNetworkType':1, #0=RNN, 1=FFN, 2=CNN
                    'learningRate': 1e-3, 
                    'dataLoaderShuffle': True,
                    'batchSize':512*2,
                    'storeModelName':storeModelName,
                    'modelName':simModel.GetModelName(),
                    'dataFile':dataFile,
                    'inputOutputSizeNN': simModel.GetInputOutputSizeNN(), 

                    'computeDevice': 'cuda' if useCUDA else 'cpu', 
                    'verboseMode': 1, 
                    'useErrorEstimator': useErrorEstimator, 
                    'maxEpochsEst': 500, 
                    'mapErrorInfo': {'eMin': -4, 'eMax': 0, 'type': 2, 'eRange': 4}, # used for log mapping of error estimator
                    'outputScalingEst': simModel.GetOutputScaling(), 
                    } # additional parameters
    
    
    parameters = { 'hiddenLayerStructureType': [0,12, 29],
                   'case': [0,1,2], 
                  }
    tStart = time.time()
    print('starting training. Please note that for multiprocessing on some machine no intermediate output is shown.')

    [parameterDict, valueList] = ParameterVariation(parameterFunction=ParameterFunctionTraining, 
                                                   parameters=parameters,
                                                    useMultiProcessing=True,
                                                   # numberOfThreads=4,
                                                 resultsFile=resultsFile+'.txt', 
                                                 addComputationIndex=True,
                                                 parameterFunctionData=functionData)
    values, nnModels, nnModelsEst = [], [], []
    for val in valueList: 
        values += [val[0]]
        nnModels += [val [1]]
        if functionData['useErrorEstimator']: 
            nnModelsEst += [val[2]]
        else: 
            nnModelsEst += [None]
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
    #check specific solution
    nntc.LoadNNModel(storeModelName+str(5)+'.pth') #best model: #5, 2500 epochs, lr0.001, 'L', NT512, HL128
    # plotTEsts and training are the indices of the shown data in the set
    MSE = nntc.EvaluateModel(plotTests=[868, 987, 1003], plotTrainings=[0,1,2,3,4], plotVars=['time','ODE2'], 
                             flagErrorBarPlot=False, nTrainingMSE = 128)
    

    #%%  
    def testModelRecursively(nnModel, simModel, iPasses, nnModelEst = None): 
        factorTest  = iPasses-1
        nTotal = simModel.nStepsTotal
        tEnd = simModel.endTime
        nOut = simModel.nOutputSteps
        
        
        h = tEnd/nTotal
        tEndTest = tEnd + h*nOut *factorTest
        nTotalTest = nTotal + nOut * factorTest
        # nOut = 20
        iterationsTest = int(np.ceil((nTotalTest - (nTotal - nOut)) / nOut))
        
        simModelTest = NonlinearOscillatorContinuous(nStepsTotal=nTotalTest, nOutputSteps = nTotalTest, endTime=tEndTest, 
                     useInitialValues=simModel.useInitialValues, 
                     frictionForce=simModel.frictionForce, nMasses=simModel.nMasses,  scalBaseMotion = simModel.scalBaseMotion, 
                     nonlinearityFactor = simModel.nonlinearityFactor, #factor for cubic nonlinearity
                     initUfact = simModel.initUfact, initVfact = simModel.initVfact, useVelocities = simModel.useVelocities)
        
        simModelTest.CreateModel()
        t = np.linspace(h, tEndTest, nTotalTest)
        
        vecTest = simModelTest.CreateInputVector()
        # vecTest = np.array([np.sin(t*50)]).T
        
        if False: # for example in the manuscript the input data is systematically changed
            vecTest[nTotal + 1*nOut : nTotal+2*nOut] = 0
            vecTest[nTotal + 2*nOut : nTotal+3*nOut] += 0.2 # constnt offset
            vecTest[nTotal + 3*nOut : nTotal+4*nOut] *= 2.5
        solRef = simModelTest.ComputeModel(vecTest)
        estErr, refErr = [], []
        
        outputNN = torch.zeros([1, nTotal-nOut, 1]) # zero solution to be overwritten
        for i in range(iterationsTest): 
            iStart = i*nOut
            iEnd = iStart + nTotal 
            flagA = iEnd > nTotalTest
            if flagA: # current solution would "stand over it"
                iDiff = iEnd - nTotalTest
                iStart -= iDiff 
                iEnd = iStart + nTotal
            outRef = solRef[iEnd-nOut:iEnd]

                
            nnInp = torch.tensor(np.array([vecTest[iStart:iEnd]]), dtype=torch.float32)   
            buffer = nnModel(nnInp)
            
            if flagA: # correctly "assemble" solution
                outputNN = torch.concatenate((outputNN, buffer[:,iDiff:]), 1)
            else: 
                outputNN = torch.concatenate((outputNN, buffer), 1)
                
            
            if not(nnModelEst is None): 
                estOut = float(nnModelEst(nnInp))
                errEstimated = mapErrorInv(estOut, nntc.nnModelEst.mapErrorInfo)
                estErr += [errEstimated]
                refErr += [torch.mean(torch.abs(buffer.detach()[0] - outRef), 0).numpy()[0]]

        outputNN_npy = torch.squeeze(outputNN).detach().numpy()
                
        plt.figure('testRecursively', figsize=(12, 6), dpi=100)
        plt.plot(t, solRef)
        plt.plot(t, outputNN_npy, '--')

        plt.xlabel('t in s')
        plt.ylabel('x in m')
        plt.grid(True)
        
        yBorders = [np.min(solRef)*1.1, np.max(solRef)*1.1]
        for i in range(iterationsTest + 1): 
            plt.plot([tEnd+ (i-1)*h*nOut]*2, yBorders, 'k--')
        
        for i in range(iterationsTest): 
            str1 = str(roundSignificant(estErr[i], 2))
            str2 = str(roundSignificant(refErr[i], 2))
            strErrorEstimator = '$\hat{e}' + '={}$, \n$e$={}'.format(str1,  str2)
            plt.text(tEnd+ (i-0.5)*h*nOut, yBorders[0], strErrorEstimator, ha='center', fontsize=12)
            
        if simModel.nonlinearityFactor ==  0: 
            plt.title('linear System')
        else: 
            plt.title(r'nonlinear: $\ddot{x} + d\dot{x} + kx + \alpha k x^3 = F$, $\alpha = ' + str(simModel.nonlinearityFactor) + '$')
        
    mapErrorInfo = functionData['mapErrorInfo']
    testModelRecursively(nntc.nnModel, simModel, 5, nntc.nnModelEst)

    
    
    

#