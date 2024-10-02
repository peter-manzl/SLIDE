#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#           03_SliderCrankFlexible 
# Details:  File for creating and learning the flexible's slider-crank deflection 
#
# Author:   Peter Manzl, Johannes Gerstmayr
# Date:     2024-04-08
# Copyright: See Licence.txt
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import exudyn as exu
from exudyn.utilities import *
# from exudyn.signalProcessing import GetInterpolatedSignalValue
from exudyn.processing import ParameterVariation

from simModels import SimulationModel, SliderCrank, AccumulateAngle
from AISurrogateLib import * #MyNeuralNetwork, NeuralNetworkTrainingCenter, PVCreateData, NeuralNetworkStructureTypes, VariableShortForm, ExtendResultsFile, ParameterFunctionTraining
                   
from exudyn.plot import PlotSensor
import AISurrogateLib as aiLib

import sys
import time
import numpy as np
# #from math import sin, cos, sqrt,pi
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
torch.set_num_threads(1)

useCUDA = torch.cuda.is_available()
# useCUDA = False #CUDA support helps for fully connected networks > 256

if __name__ == '__main__': #include this to enable parallel processing
    print('pytorch cuda=',useCUDA)

# currently 12 possible activation types
def GetActivationType(iAct): 
    activationFuncions = [nn.ReLU, nn.ELU, nn.GELU, nn.GLU, nn.Tanh, nn.CELU, nn.CELU,
                          nn.Hardsigmoid, nn.Hardtanh, nn.SiLU, nn.Mish, nn.LeakyReLU]
    return activationFuncions[iAct]


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#put this part out of if __name__ ... if you like to use multiprocessing

parameterVariation = True

endTime=4
useFriction=False
nStepsTotal= 128 #96
nOut = 32

t_diff = endTime * (nStepsTotal - nOut) /nStepsTotal


simModel = SliderCrank(nStepsTotal=nStepsTotal, nOutputSteps = nOut, tStartup = 1, endTime= endTime, 
                              useInitialVelocities=False, useInitialAngles=True, 
                              useTorqueInput=False, flagFlexible=True,  useVelocityInput = False, 
                              flagVelNoise = True, trajType = 0, usePosInput=True, outputType=1, 
                              initAngleRange = [-np.pi,np.pi], 
                              vMax = 8, aMax = 20)


nnModel = MyNeuralNetwork(inputOutputSize = simModel.GetInputOutputSizeNN(), # input and output size, 
                           neuralNetworkTypeName = 'FFN',
                           hiddenLayerSize = 40,
                           hiddenLayerStructure = ['L', 'R', 'L'],
                           computeDevice='cpu',
                           # typeNumber = 1, 
                           # activationType = 1
                           ) 

# print('input size: ', simModel.GetInputOutputSizeNN())
#MyNeuralNetwork()
nntc = NeuralNetworkTrainingCenter(nnModel, simulationModel=simModel, computeDevice='cuda' if useCUDA else 'cpu',
                                   verboseMode=0, nnModelEst = nnModel)
aiLib.moduleNntc = nntc #this informs the module about the NNTC (multiprocessing)

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++{}+++++++++++++++++++++++
#in this section we create or load data
nTraining= int(1024*20) # 4096*2 # 1024*2 # 1024 * 2#  1024 *2
nTest= int(1024*2)  # 640 # 256*4 # 64*2 # 256 #64*2 #64*2 # 64*2


    
dataFile = 'data/SliderCrank' + 'flexible'*bool(simModel.flagFlexible)+ 'T'+str(nTraining)+'-'+str(nTest)+'s'+\
            str(nStepsTotal)+'ang'+str(int(simModel.useInitialAngles)) +'_' +str(round(simModel.initAngleRange[1],3))+'vel'+str(int(simModel.useInitialVelocities))+\
                str(simModel.initVelRange[1]) +'t'+str(endTime) + '_flexible'*bool(simModel.flagFlexible) + \
                    'nOut'+str(simModel.nOutputSteps) + '_TrajType'+str(simModel.trajType) + 'noise'*bool(simModel.flagVelNoise) \
                    + 'inpCut' + str(simModel.nCutInputSteps) + '_outputType'+str(simModel.outputType) + 'Est'*bool(nntc.useErrorEstimator)\
                    + 'dEI_' + "{:.1e}".format(simModel.damping[0]) + "_dEA_" + "{:.1e}".format(simModel.damping[1])
createData = aiLib.checkCreateData(dataFile)
# createData = True  # can be overwritten to generate new training and test data sets. 

if __name__ == '__main__': #include this to enable parallel processing
    if createData:
        nntc.verboseMode = 1
        # for i in 
        print('start data creation')
        nntc.CreateTrainingAndTestData(nTraining=nTraining, nTest=nTest,
                                        parameterFunction=PVCreateData, #for multiprocessing
                                       #showTests=[0,1], #run SolutionViewer for this test
                                       )
        nntc.SaveTrainingAndTestsData(dataFile)
        sys.exit()

if not createData and not parameterVariation:
    nntc.LoadTrainingAndTestsData(dataFile, nTraining=64, nTest=20)
    inpTest = torch.tensor([nntc.inputsTest[0]], dtype=torch.float32)   # size of input tensor is [1, 120, 6]; 
                                                                        # first dimenstion is the size
    nnModelEst(inpTest) # just to the neural network is. 
# #%%
# #check created training or test data:
# for j in range(50):
#     data=nntc.GetNNModel().OutputData2PlotData(nntc.targetsTraining[j])
#     nntc.GetNNModel().mbs.PlotSensor([data],components=[1], newFigure=j==0)
# #%%

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__': #include this to enable parallel processing
    
    identifierString = 'Deflection_F6'
    storeModelName = 'model/'+simModel.GetModelName()+identifierString
    resultsFile = 'solution/res_'+simModel.GetModelNameShort()+'Res'+identifierString
    #%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if parameterVariation:
        functionData = {'maxEpochs': 1500, # 400, # 1200, # 10000, # 10_000,
                        'nTraining': 1024*20, #1024, # 1024*4,
                        'nTest': 1024*2,
                        'lossThreshold':1e-10,
                        'lossLogInterval':10,
                        'testEvaluationInterval':25, # or 100?
                        'learningRate': 1.5e-3, 
                        # 'resLayerList': [], 
                        # 'hiddenLayerSize':128,
                        'hiddenLayerStructureType': 12,
                        # 'activationType': 0, '
                        'storeModelName':storeModelName,
                        'modelName':simModel.GetModelName(),
                        'dataFile':dataFile,
                        'inputOutputSizeNN': [(2, 96), (32,1)], 
                        # 'simulationModel': simModel, 
                        # 'hiddenLayerStructureType': 26, 
                        'computeDevice': 'cuda' if useCUDA else 'cpu', 
                        'verboseMode': 1, 
                        # 'customNetwork': CustomNNSliderCrank, # here a custom neural network could be implemented
                        # 'customNetworkEst': CustomNNSliderCrank, 
                        'typeNumber': 5, 
                        'verboseMode': 1,
                        'useErrorEstimator': True, 
                        'maxEpochsEst': 1000, 
                        'mapErrorInfo': {'eMin': -3, 'eMax': 0, 'type': 2, 'eRange': 3},
                        'outputScalingEst': simModel.GetOutputScaling(), 
                        } #additional parameters
        
        functionData['batchSize'] = functionData['nTraining'] // 8        
        parameters = { 
                       'hiddenLayerSize': [256], #[192],
                        # 'case': [0,1], # changing seeeds
                      }
        
        tStart = time.time()
        numRuns = aiLib.GetNumOfParameterRuns(parameters)
        if numRuns == 1 and not(useCUDA): 
            nThreadsRun = os.cpu_count()-2
            torch.set_num_threads(nThreadsRun)
            print('set torch number of threads to: ', nThreadsRun)
        flagRunTraining = checkForModel(resultsFile)
        if flagRunTraining: 
            [parameterDict, valueList] = ParameterVariation(parameterFunction=ParameterFunctionTraining, 
                                                            parameters=parameters,
                                                            useMultiProcessing=(numRuns>1),
                                                          # numberOfThreads=4,
                                                         resultsFile=resultsFile+'.txt', 
                                                         addComputationIndex=True,
                                                         parameterFunctionData=functionData)
            values, nnModels = [], []
            for val in valueList: 
                values += [val[0]]
                nnModels += [val [1]]
            CPUtime=time.time()-tStart
            print('training variation took:',round(CPUtime,2),'s')
            # print('values=', values)
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
        nntc.PlotTrainingResults(resultsFileNpy, dataFile, plotTestsMeanError=True)
        
        
    
        #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #check specific solution
        numModels = 0
        for savedModels in os.listdir('model'): 
            if storeModelName.split('/')[1] in savedModels and not('_estimator' in savedModels): 
                numModels += 1
        for i in range(numModels): 
            nntc.LoadNNModel(storeModelName+str(i)+'.pth') #best model: #5, 2500 epochs, lr0.001, 'L', NT512, HL128
            values = nntc.EvaluateModel(plotTests=[0,1,2,3,4], plotTrainings=[0,1,2,3,4], plotVars=['time','x'], computeRMSE = True)
            print('mean RMSE: ', np.mean(values['testRMSE']))
            print('mean MSE: ', np.mean(values['testMSE']), '\n')
            

        nntc.LoadNNModel(storeModelName+str(0)+'.pth') #best model: #5, 2500 epochs, lr0.001, 'L', NT512, HL128
        # nntc.EvaluateModel(plotTests=[0,1,2,3,4], plotTrainings=[0,1,2,3,4], plotVars=['time','x'])
        values = nntc.EvaluateModel(plotTests=[0,1,2,3,4], plotTrainings=[0,1,2,3,4, 5], plotVars=['time','x'], flagErrorBarPlot=True)
        nntc.EvaluateModelEstimator(64, False, True, factorUnit=1)
        #%% 
        fig, axs = plt.subplots(2)
        for i in range(5): 
            phiTest = AccumulateAngle(np.arctan2(nntc.inputsTest[i][1], nntc.inputsTest[i][0]))
            # plt.plot(phiTest)
            axs[0].plot(phiTest)
            axs[1].plot(np.diff(phiTest) / (endTime/nStepsTotal))
        for i in range(2): 
            axs[i].grid()
            # axs[i].legend()
    #%% 
    aiLib.FinishSound()
    inp = torch.tensor([nntc.inputsTest[0]], dtype=torch.float32)
    def compute(mySimModel): 
        vec = mySimModel.CreateInputVector()
        out = mySimModel.ComputeModel(vec)
        return out

    def testModelSLIDE(nnModel, simModel, iPasses, nnModelEst): 
     #%% 
        factorTest  = iPasses-1
        nTotal = simModel.nStepsTotal
        tEnd = simModel.endTime
        nOut = simModel.nOutputSteps
        
        h = tEnd/nTotal
        tEndTest = tEnd + h*nOut *factorTest
        nTotalTest = nTotal + nOut * factorTest
        iStartup = int(simModel.tStartup / h)
        factorMM = 1e3  
        np.random.seed(44)
        iterationsTest = int(np.ceil((nTotalTest - (nTotal - nOut)) / nOut))
        

        simModelTest = SliderCrank(nStepsTotal=nTotalTest, nOutputSteps = nTotalTest, tStartup = 1, endTime= tEndTest, 
                                          useInitialVelocities=False, useInitialAngles=True, 
                                          useTorqueInput=False, flagFlexible=True,  useVelocityInput = False, 
                                          flagVelNoise = False, trajType = 0, usePosInput=True, outputType=1, 
                                          initAngleRange = [-np.pi,np.pi], # , initVelRange  = [-12,12],
                                          vMax = 8, aMax = 20)
        simModelTest.CreateModel()
        
        t = np.linspace(h, tEndTest, nTotalTest)
        vecTest = simModelTest.CreateInputVector()
        phiInp = AccumulateAngle(np.arctan2(vecTest[1,:], vecTest[0,:])) # could be used for plotting
        
        # AccumulateAngle
        solRef = simModelTest.ComputeModel(vecTest)
        outputScaling = simModelTest.GetOutputScaling()
          
#%%         
        testMSEEst, estErr, refErr = [], [], []
        outputNN = torch.zeros([1, nTotal-nOut, 1])# torch.tensor([], dtype=torch.float32)
        for i in range(iterationsTest): 
            iStart = (i)*nOut + simModelTest.nCutInputSteps
            iEnd = iStart + nTotal  - simModelTest.nCutInputSteps
            flagA = iEnd > nTotalTest
            if flagA: 
                iDiff = iEnd - nTotalTest
                iStart -= iDiff #iEnd - nTotalTest
                iEnd = iStart + nTotal
                    
            outRef = solRef[iEnd-nOut:iEnd]
            nnInp = torch.tensor(np.array([vecTest[:,iStart:iEnd]]), dtype=torch.float32).to(nntc.computeDevice)   
            buffer = nnModel(nnInp) # model output of current segment
             
            if flagA: 
                outputNN = torch.concatenate((outputNN, buffer[:,iDiff:].cpu()), 1)
            else: 
                outputNN = torch.concatenate((outputNN, buffer.cpu()), 1)
                
            if not(nnModelEst is None): 
                estOut = float(nnModelEst(nnInp))
                errEstimated = mapErrorInv(estOut, nntc.nnModelEst.mapErrorInfo)
                estErr += [errEstimated]
                err_i = torch.mean(torch.abs(buffer.cpu().detach()/ outputScaling[:nOut,:] - outRef / outputScaling[:nOut,:]), 1)
                refErr += [err_i.numpy().reshape(1)[0]]
                print('segment {}: {}% estimator accuracy'.format(i, round(100*refErr[-1]/ np.mean(np.linalg.norm((buffer.cpu().detach().numpy()/ outputScaling[:nOut,:]), axis=2)), 2)))
                    
            outputNN_npy = torch.squeeze(outputNN).cpu().detach().numpy()
            
        outputNN_npy = torch.squeeze(outputNN).detach().numpy()
        
        plt.figure('testRecursively')
        plt.gcf().set_size_inches(13,5) # size: 13 x 5 inches
        plt.plot(t, solRef/20, label='reference')
        plt.plot(t, outputNN_npy/20, '--', label='NN solution')

        plt.xlabel('t in s')
        plt.ylabel('deflection $d$ in $mm$')
        plt.grid(True)
        
        yBorders = [np.min(solRef), np.max(solRef)]
        lim = plt.gca().get_ylim()
        lim = (-0.042, 0.061)
        for i in range(iterationsTest + 1):     
            plt.plot([tEnd+ (i-1)*h*nOut]*2, yBorders, 'k--')
            
        for i in range(iterationsTest): 
            str1 = str(roundSignificant(estErr[i]*factorMM, 2))
            str2 = str(roundSignificant(refErr[i]*factorMM, 2))
            strErrorEstimator = '$\hat{e}' + '={}$, \n$e={}$'.format(str1,  str2)
            print('segment {}: est {}% accuracy'.format(i, round(100*(estErr[i] - refErr[i])/refErr[i], 2)))
            plt.text(tEnd+ (i-0.5)*h*nOut, lim[0]*0.94, strErrorEstimator, ha='center', fontsize=15)


        yTicks = plt.gca().get_yticks()
        plt.yticks(yTicks, np.round(yTicks*factorMM, 10))
        plt.ylim([-0.042, 0.061])   
        plt.xlim([-0.05, t[-1]*1.01])
        plt.tight_layout()
    #%% 
    
    testModelSLIDE(nntc.nnModel, simModel, 8, nntc.nnModelEst)
    plt.waitforbuttonpress()


