#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#           04_Flexible_6RRobot
# Details:  File for creating data and learning the the socket's deflection of 
#           a robot standing on the flexible socket. 
#
# Author:   Peter Manzl, Johannes Gerstmayr
# Date:     2024-10-01
# Copyright: See Licence.txt
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import exudyn as exu
from exudyn.utilities import *
# from exudyn.signalProcessing import GetInterpolatedSignalValue
from exudyn.processing import ParameterVariation


from simModels import Flex6RRobot
from AISurrogateLib import * #MyNeuralNetwork, NeuralNetworkTrainingCenter, PVCreateData, NeuralNetworkStructureTypes, VariableShortForm, ExtendResultsFile, ParameterFunctionTraining
import AISurrogateLib as aiLib
from exudyn.plot import PlotSensor


import sys
import time
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
torch.set_num_threads(1) # if parallelization is desired, 

useCUDA = torch.cuda.is_available()

# torch.set_num_threads(10)
if __name__ == '__main__': #include this to enable parallel processing
    print('pytorch cuda=',useCUDA)



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#put this part out of if __name__ ... if you like to use multiprocessing

endTime=2
nStepsTotal=120
nOutput = 60
isRigid = False

class CustomNNRobot(MyNeuralNetwork):
    def __init__(self, inputOutputSize = None, hiddenLayerSize = None, hiddenLayerStructure = ['L'], 
                 neuralNetworkTypeName = 'RNN', computeDevice = 'cpu', 
                 rnnNumberOfLayers = 1, resLayerList = [], hiddenLayerSizeList = [],
                 rnnNonlinearity = 'tanh', activationType = 0, typeNumber = 3, #'relu', None; 'tanh' is the default
                 ):
        super().__init__(inputOutputSize = inputOutputSize, hiddenLayerSize = hiddenLayerSize, hiddenLayerStructure = hiddenLayerStructure, 
                      neuralNetworkTypeName = neuralNetworkTypeName, computeDevice = computeDevice, 
                      rnnNumberOfLayers = rnnNumberOfLayers,rnnNonlinearity = rnnNonlinearity)
        self.myNN = self.customNetworks(inputOutputSize, hiddenLayerSize, typeNumber = typeNumber, resLayerList= resLayerList, activationType=activationType, computeDevice=computeDevice)
    
    # create cusotom networks depending on activationType and typenumber
    def customNetworks(self, inputOutputSize, hiddenLayerSize, typeNumber = 0, activationType = 0, resLayerList = [], computeDevice = 'cpu'): 
        activationType  = GetActivationType()[activationType]

        self.typeNumber  = typeNumber 
        self.resLayerList = resLayerList
        nnList = []        
        
        # type 0 uses 3 parallel nets for x,y,z deflections
        if typeNumber  == 0: 
            # this part of the model is the same for everybody!
            nInTotal = np.prod(inputOutputSize[0])
            nOutTotal = np.prod(inputOutputSize[1])
            nOutSplit = nOutTotal //3 
            self.nInTotal, self.nOutTotal, self.nOutSplit = nInTotal, nOutTotal, nOutSplit
            model = nn.ModuleList([nn.Flatten(1), 
                                   # nn.Linear(nInTotal, nInTotal)
                                   ])
            
            # modelSplit = []
            for i in range(3): # 3 outputs: x,y,z! 
                activationType = nn.ELU # fixed to ELU
                modelSplit = nn.Sequential(nn.Linear(nInTotal, hiddenLayerSize))
                # modelSplit.append(nn.Linear(nInTotal//3, hiddenLayerSize))
                # model.append(nn.Linear(np.prod(inputOutputSize[0]), hiddenLayerSize))
                modelSplit.append(activationType())
                modelSplit.append(nn.Linear(hiddenLayerSize, hiddenLayerSize))
                # modelSplit.append(nn.Linear(hiddenLayerSize, hiddenLayerSize))
                modelSplit.append(activationType())
                if True: 
                    modelSplit.append(nn.Linear(hiddenLayerSize, hiddenLayerSize))
                    # modelSplit.append(nn.Linear(hiddenLayerSize, hiddenLayerSize))
                    modelSplit.append(activationType())
                modelSplit.append(nn.Linear(hiddenLayerSize, nOutSplit))
                modelSplit.to(computeDevice)
                # modelSplit.append(nn.Unflatten(-1, inputOutputSize[1]))
                model.append(modelSplit)
            # self.nLayers = 2
            # self.nLayersSplit = 9
            model.append(nn.Unflatten(-1, inputOutputSize[1]))
            model.to(computeDevice)
            self.xOut = None
            
        elif typeNumber == 1: 
            activationType = nn.Tanh
            model = nn.Sequential(nn.Flatten(1))
            model.append(nn.Linear(np.prod(inputOutputSize[0]), hiddenLayerSize))
            model.append(nn.Tanh())
            model.append(nn.Linear(hiddenLayerSize, hiddenLayerSize))
            model.append(activationType())
            model.append(nn.Linear(hiddenLayerSize, hiddenLayerSize))
            model.append(activationType())
            model.append(nn.Linear(hiddenLayerSize, hiddenLayerSize))
            model.append(activationType())
            model.append(nn.Linear(hiddenLayerSize, hiddenLayerSize))
            model.append(activationType())
            model.append(nn.Linear(hiddenLayerSize, hiddenLayerSize))
            model.append(activationType())
            model.append(nn.Linear(hiddenLayerSize, np.prod(inputOutputSize[1])))
            if exu.advancedUtilities.IsInteger(inputOutputSize[1]): 
                model.append(nn.Unflatten(-1, [inputOutputSize[1]]))
            else: 
                model.append(nn.Unflatten(-1, inputOutputSize[1]))
            
        print('custom network: \n{}'.format(model))
        return model 
    
    def forward(self, x): 
        if self.typeNumber == 0: 
            xIn_split = self.myNN[0](x)
            if self.xOut is None or self.xOut.shape !=[x.shape[0],self.nOutTotal]:     
                self.xOut = torch.zeros([x.shape[0],self.nOutTotal]).to(self.computeDevice)
            iOffset = 0
            for i in range(3): 
                xOut_split = self.myNN[i+1](xIn_split)
                self.xOut[:,iOffset:iOffset+self.nOutSplit] = xOut_split
                iOffset += self.nOutSplit
            return self.myNN[-1](self.xOut)

            
            
        if  self.typeNumber == 1 or self.typeNumber == 2: 
            return x.view(-1, nnModel.myNN[-1].out_features, 1)
        # if self.typeNumber == 0: 
            # pass 
            # todo: concat data! 
            # torch.cat()
            # return x.view(-1, nnModel.myNN[-1].out_features, 1)
        
        return x
    
simModel = Flex6RRobot(nStepsTotal=nStepsTotal, endTime=endTime, 
                 isRigid=isRigid, createModel=False, verboseMode=0, inputType=4, outputType = 2, 
                 nOutputSteps = nOutput, EModulus = 1e9)
simModel.CreateModel([0]*6, flagComputeModel=True)

def compute(mySimModel): 
    vec = mySimModel.CreateInputVector()
    out = mySimModel.ComputeModel(vec)
    return out

# model.inputStep = True #for n masses
nnModel = MyNeuralNetwork(inputOutputSize = simModel.GetInputOutputSizeNN(), # input and output size, 
                           neuralNetworkTypeName = 'FFN',
                           hiddenLayerSize = 40,
                           hiddenLayerStructure = ['L', 'R', 'L'],
                           computeDevice='cpu',
                           # typeNumber = 1, 
                           # activationType = 1
                           )
nnModel2 = CustomNNRobot(inputOutputSize = simModel.GetInputOutputSizeNN(), # input and output size, 
                           neuralNetworkTypeName = 'FFN',
                           hiddenLayerSize = 40,
                           hiddenLayerStructure = ['L', 'R', 'L'],
                           computeDevice='cpu',
                            typeNumber =0, 
                            activationType = 1
                           )

#MyNeuralNetwork()
nntc = NeuralNetworkTrainingCenter(nnModel, simulationModel=simModel, computeDevice='cuda' if useCUDA else 'cpu',
                                   verboseMode=0, nnModelEst = nnModel)
aiLib.moduleNntc = nntc #this informs the module about the NNTC (multiprocessing)


#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#in this section we create or load data
nTraining= 1024*20 # original: 20480
nTest = 4096 


    
dataFile = 'data/FFRF6RRobot_T'  +str(nTraining)+'-'+str(nTest)+'st'+\
            str(nStepsTotal) + 'stOut' + str(simModel.nOutputSteps) + 'te' + str(simModel.endTime) \
                + '_ErrorEstimator'*bool(nntc.useErrorEstimator) + '_inputType' + str(simModel.inputType) \
                + '_outputType' + str(simModel.outputType)\
                + 'EModulus' + "{:.1e}".format(simModel.EModulus) + "D" + "{:.1e}".format(simModel.dampingK)


createData = aiLib.checkCreateData(dataFile)
# createData = True  # can be overwritten to generate new training and test data sets. 

if __name__ == '__main__': #include this to enable parallel processing
    if createData:
        nntc.verboseMode = 1
        if nntc.useErrorEstimator: 
            print('\ncreate datafile INCLUDING error estimator. Please note that for some models this may take a while. creating file:')
            print(dataFile)
            print('including a total of {} simulation runs'.format(nTraining*2+nTest*2))
            # using a i9 with 20 threads this takes approximatly 2 - 3 hours. 
            # On the cluster this can be accelerating greatly using SLURM and MPI. 
            # because the data is created independently good scaling is observed. 
        nntc.CreateTrainingAndTestData(nTraining=nTraining, nTest=nTest,
                                        parameterFunction=PVCreateData, # to toggle multiprocessing acomment (off)
                                       #showTests=[0,1], #run SolutionViewer for this test to visualize
                                       )
        nntc.SaveTrainingAndTestsData(dataFile)
        sys.exit()

if not createData and False: # for testing with real data
    nntc.LoadTrainingAndTestsData(dataFile, nTraining=128, nTest=32)

    inpTest = torch.tensor([nntc.inputsTest[0]], dtype=torch.float32)   # size of input tensor is [1, 120, 6]; 
                                                                        # first dimenstion is the size
    # nnModel2(inpTest) # just to the neural network is. 


#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__': #include this to enable parallel processing
    try: 
        nThreads = os.environ['SLURM_NTASKS'] 
        flagMPI = True
        # SLURM is running on the cluster "Leo5", thus we can detect if we work on a cluster on local machine. 
        # for running simulations and/or training on the cluster MPI (Message Passing Interface) is used. 
    except: 
        nThreads = os.cpu_count()
        flagMPI = False
    print('start variation; multiprocessing using {} threads'.format(nThreads), 'using MPI'*flagMPI)
    
    identifierString = 'Estimator_G5'
    storeModelName = 'model/'+simModel.GetModelName()+identifierString
    # useErrorEstimator = False
    resultsFile = 'solution/res_'+simModel.GetModelNameShort()+'Res'+identifierString
    # %%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    functionData = {'maxEpochs': 2000, #1_000, # 10_000,
                    'nTraining':1024*20,
                    'nTest': 1024*4,
                    'lossThreshold':1e-6,
                    'lossLogInterval':25,
                    'testEvaluationInterval': 25, # or 100?
                    'learningRate': 1.5e-3, 
                    'epochPrintInterval': 25,
                    # 'hiddenLayerStructureType': 12,
                    # 'batchSize': 512*2,
                    'storeModelName':storeModelName,
                    'modelName':simModel.GetModelName(),
                    'dataFile':dataFile,
                    'inputOutputSizeNN': simModel.GetInputOutputSizeNN(), 
                    'customNetwork': CustomNNRobot, 
                    'computeDevice': 'cuda' if useCUDA else 'cpu', 
                    'verboseMode': 1, 
                    'useErrorEstimator': True, 
                    'maxEpochsEst': 800, 
                    'outputScalingEst': simModel.GetOutputScaling(), 
                    'mapErrorInfo': {'eMin': -4.5, 'eMax': -1.5, 'type': 2, 'eRange': 3},
                    } #additional parameters
    
    functionData['batchSize'] = functionData['nTraining']//8
    # attention: multiprocessing currently does not work with CUDA
    parameters = { 'hiddenLayerStructureType': [26], 
                    'hiddenLayerSizeEst': [360], 
                    'hiddenLayerSize': [180], 
                    'neuralNetworkType': [1],
                    "case": [0], # initialize seed 0
                  }
    
    tStart = time.time()
    if 'useErrorEstimator' in functionData.keys() and functionData['useErrorEstimator'] == True: 
        resultsFile += '_errorEstimator'
        useErrorEstimator  = True
    numRuns = 1
    for key, value in parameters.items(): 
        print(key)
        numRuns *= len(value)

    flagRunTraining = checkForModel(resultsFile) # if neural network already exists, ask if it should be overwritten. 
    if flagRunTraining: 
        [parameterDict, valueList] = ParameterVariation(parameterFunction=ParameterFunctionTraining, 
                                                  parameters=parameters,
                                                    useMultiProcessing=(numRuns > 1),
                                                  numberOfThreads=nThreads, 
                                                  resultsFile=resultsFile+'.txt', 
                                                  addComputationIndex=True,
                                                  parameterFunctionData=functionData, 
                                                  useMPI=flagMPI)
        values, nnModels, nnModelEst = [], [], []
        for val in valueList: 
            values += [val[0]]
            nnModels += [val [1]]
            if 'errorEstimator' in resultsFile: 
                nnModelEst += [val[2]]
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
            np.save(f, dataDict, allow_pickle=True) # allow_pickle=True for lists or dictionaries

    #%%++++++++++++++++++++++++++++++++++++++
    #show loss, test error over time, ...
    resultsFileNpy = resultsFile+'.npy'    
    nntc.PlotTrainingResults(resultsFileNpy, dataFile, plotTestsMeanError=True)
    
    
    
    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #check specific solution
    nntc.LoadNNModel(storeModelName+str(0)+'.pth') # load model number 0 from the training. 
    compShow = ['posArmX','posArmY', 'posArmZ'] 
    for comp in compShow: # only visualize the data from compShow
        val = nntc.EvaluateModel(plotTests=[0,1,2,3,4], plotTrainings=[0,1,2,3,4], plotVars=['time',comp], computeRMSE=True)

    # evaluation of error estimator; 
    nntc.EvaluateModelEstimator(nMax = 64, flagPlotBestWorst =True, 
                                flagPlotCorreleation=True, factorUnit=1e3)

    
        
    #%% call model recursively for application of SLIDE 
    def testModelSLIDE(nnModel, simModel, iPasses, nnModelEst = None): 
        factorTest  = iPasses-1
        np.random.seed(42)
        nTotal = simModel.nStepsTotal
        tEnd = simModel.endTime
        nOut = simModel.nOutputSteps
        scaleToMM = 1e3 # scaling factor to millimetres
        
        h = tEnd/nTotal
        tEndTest = tEnd + h*nOut *factorTest
        nTotalTest = nTotal + nOut * factorTest
        tTraj =  nTotal*h/2

        iterationsTest = int(np.ceil((nTotalTest - (nTotal - nOut)) / nOut))
        simModelTest = Flex6RRobot(nStepsTotal=nTotalTest, endTime=tEndTest, 
                           isRigid=isRigid, createModel=False, verboseMode=0, inputType=simModel.inputType, outputType = simModel.outputType, 
                           EModulus=simModel.EModulus, 
                           )
        simModelTest.CreateModel()
        outputScaling = simModelTest.GetOutputScaling()
         

        q0 = np.random.uniform(simModelTest.qLim[0], simModelTest.qLim[1])
        qi = np.random.uniform(simModelTest.qLim[0], simModelTest.qLim[1])
        myTraj = Trajectory(q0)
        myTraj.Add(ProfileConstantAcceleration(qi, tTraj))
        for i in range(iterationsTest*2):
             qi = np.random.uniform(simModelTest.qLim[0], simModelTest.qLim[1]) # (np.random.random(6)-0.5) * 2*np.pi
             myTraj.Add(ProfileConstantAcceleration(qi,tTraj))
             print(qi)

        vecTest = np.zeros([nTotalTest,6])
        q_axs0 = []
        for i, t in enumerate(simModelTest.timeVecIn):
             q, v, a = myTraj.Evaluate(t)
             vecTest[i,:] = q  

        vecTest /= np.pi
        solRef = simModelTest.ComputeModel((vecTest, myTraj), flagRenderer=False)

        outputNN = torch.zeros([1, nTotal-nOut, 3], dtype=torch.float32).to('cuda')
        estErr, refErr = [], []
        for i in range(iterationsTest): 
            iStart = i*nOut
            iEnd = iStart + nTotal 
            flagA = iEnd > nTotalTest
            if flagA: 
                iDiff = iEnd - nTotalTest
                iStart -= iDiff # iEnd - nTotalTest
                iEnd = iStart + nTotal
                    
            outRef = solRef[iEnd-nOut:iEnd]
            nnInp = torch.tensor(np.array([vecTest[iStart:iEnd]]), dtype=torch.float32).to('cuda')   
            buffer = nnModel(nnInp) # model output of current segment
             
            if flagA: 
                outputNN = torch.concatenate((outputNN, buffer[:,iDiff:]), 1)
            else: 
                outputNN = torch.concatenate((outputNN, buffer), 1)
                
            if not(nnModelEst is None): 
                estOut = float(nnModelEst(nnInp))
                errEstimated = mapErrorInv(estOut, nntc.nnModelEst.mapErrorInfo)
                estErr += [errEstimated]
                err_i = torch_rmse(buffer.cpu().detach()/ outputScaling[:nOut,:] - outRef/outputScaling[:nOut,:])
                refErr += [torch_rmse(err_i)[0].numpy()]
                print('{}'.format(round(100*refErr[-1]/ np.mean(np.linalg.norm((buffer.cpu().detach().numpy()/ outputScaling[:nOut,:]), axis=2)), 2)))
                
        outputNN_npy = torch.squeeze(outputNN).cpu().detach().numpy()

        fig, axs = plt.subplots(3)
         
        for i in range(3): 
            axs[i].plot(simModelTest.timeVecOut, scaleToMM*solRef[:,i]/outputScaling[:,i], label='reference')
            axs[i].plot(simModelTest.timeVecOut, scaleToMM*outputNN_npy[:,i]/outputScaling[:,i], '--', label='NN solution')
        plt.xlabel('t in s')
         
        yBorders = [np.min(solRef/outputScaling)*scaleToMM, np.max(solRef/outputScaling)*scaleToMM]
        yLabelList = ['x', 'y', 'z']
        for j in range(3):
            for i in range(iterationsTest + 1): 
                lim = axs[j].get_ylim()
                axs[j].plot([tEnd+ (i-1)*h*nOut]*2, np.array(yBorders)*2, 'k--')
                axs[j].grid(True)
                axs[j].set_ylabel('deflection {} in mm'.format(yLabelList[j]))
                axs[j].set_ylim(lim)
        lim = list(lim)
        lim[0] *= 1.45
        axs[2].set_ylim(lim)

        for i in range(iterationsTest): 
            str1 = str(roundSignificant(estErr[i]*scaleToMM, 2))
            str2 = str(roundSignificant(refErr[i]*scaleToMM, 2))
            strErrorEstimator = '$\hat{e}' + '={}$mm, \n$e={}$mm'.format(str1,  str2)
            print('segment {}: est {}% accuracy'.format(i, round(100*(estErr[i] - refErr[i])/refErr[i], 2)))
            axs[-1].text(tEnd+ (i-0.5)*h*nOut, lim[0]*0.965, strErrorEstimator, ha='center', fontsize=12)

        plt.gcf().set_figwidth(10)
        plt.gcf().set_figheight(8)
        plt.tight_layout()
        
    nnModel = nntc.nnModel
    nnModelEst = nntc.nnModelEst
    #%%
    testModelSLIDE(nntc.nnModel, simModel, 10, nntc.nnModelEst)
    plt.waitforbuttonpress()
    
# %% speedup tests
# test model performance on larger batches compared to CPU

flagSpeedupTests = False
if flagSpeedupTests : 
    toMilliSeconds = 1e3
    tSim = 4.76 * toMilliSeconds # in milli seconds; previously measured
    # %timeit simModel.ComputeModel(inpSim)
    # tSim = 748 # time of the slider-crank simulation 
    
    import timeit
    
    def forwardPass(nnModel, inp): 
        return nnModel(inp)
    

    # N1 = [1,10,25,50, 100, 250, 500, 1000, 2500, 5000, 10000]
    N1 = np.logspace(0, np.log10(nntc.inputsTraining.shape[0]), 50)
    N1[-1] = nntc.inputsTraining.shape[0]
    N1 = np.array(N1, dtype=int)
    N2 = [1,100,1000]
    myTest = N1
    myTest = np.sort(np.array(list(set(myTest))))
    
    data = []
    speedupRelative = []
    NRuns = 5000
    flagRunWarmup = False
    for nTest in myTest: 
        print('n=', nTest)
        nnModel = nntc.GetNNModel()
        nnModel.myNN.eval()
        if nTest == 1: 
            inp = torch.tensor([nntc.inputsTraining[0]], dtype=torch.float32).to('cuda')
        else: 
            inp = torch.tensor(nntc.inputsTraining[0:nTest], dtype=torch.float32).to('cuda')
        if flagRunWarmup: # does not make a difference here... 
            for i in range(5): forwardPass(nnModel, inp) # "run warm..." 
        data += [timeit.timeit("forwardPass(nnModel, inp)", globals=globals(), number=NRuns)/NRuns * toMilliSeconds] # *1000 to convert to ms
        speedupRelative += [tSim*nTest/data[-1] * 0.25]
        # globals needed to run in current "global" namespace; number is the number of runs
    plt.figure('timing6R')
    plt.semilogx(myTest, np.array(data)) # in ms
    plt.ylabel('runtime in ms')
    plt.xlabel('data size')
    
    plt.figure('speedup')
    plt.loglog(myTest, speedupRelative, 'o--')
    plt.ylabel('speedup')
    plt.xlabel('batchsize')
    plt.grid(True)
    plt.gcf().set_size_inches(8,4)
    plt.tight_layout()