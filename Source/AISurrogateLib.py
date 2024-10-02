#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#           AISurrogteLib
#
# Details:  Library to support creation and testing of neural networks for multibody surrogate models
#
# Author:   Peter Manzl and Johannes Gerstmayr 
# date:     2024-10-01
# Copyright: See Licence.txt
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



import exudyn as exu
from exudyn.utilities import SmartRound2String
from exudyn.signalProcessing import GetInterpolatedSignalValue
from exudyn.processing import ParameterVariation
from exudyn.plot import PlotSensor, listMarkerStyles

import os
import sys
import copy
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# plotting
import matplotlib.pyplot as plt
from matplotlib import colors as mColors

from timeit import default_timer as timer
from simModels import SimulationModel, NonlinearOscillator, SliderCrank
from exudyn.robotics.motion import Trajectory, ProfileConstantAcceleration, ProfilePTP



import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import random
import time


def FinishSound(): 
    try: 
        import winsound
    except: 
        return
    duration = int(1000 /5) # milliseconds
    freq = int(440/5)  # Hz
    for i in range(6): 
        winsound.Beep(int(freq *(i+4)), duration)
        
def checkForModel(resultsFile): 
    if resultsFile.split('/')[1] + '.npy' in os.listdir('solution'):     
        myInp = input('Do you want to overwrite the previously created model? \ny/n: ')
        if myInp in ['y', 'Y', 'yes', 'Yes', 'YES']: 
            flagRunTraining = True
        else: 
            flagRunTraining = False
    else: 
        flagRunTraining = True
    return flagRunTraining

def mapError(err, mapType=0, mapInfo = {}): 
    errLog = torch.log10(err)
    # todo: catch if err == 0 
    if len(mapInfo.keys()) == 0: 
        
        eMin = torch.min(errLog)
        eMax = torch.max(errLog)
        mapInfo = {'type': mapType, 
                   'eMin': eMin, 
                   'eMax': eMax}
    else:  # read from given Dict if possible ... 
        eMin = mapInfo['eMin']
        eMax = mapInfo['eMax']
        
    eRange = eMax - eMin
    mapInfo['mapType'] = mapType
    
    if mapType == 0: 
        x = err
    elif mapType == 1: 
        x = (errLog - eMin)/eRange
    elif mapType == 2: 
        x = (errLog + (eMax - eMin)/2)/(eRange/2)

    return x, mapInfo

def mapErrorInv(err, mapInfo = {}, mapType = None): 
    eMin= mapInfo['eMin']
    eMax = mapInfo['eMax']
    eRange = mapInfo['eRange']
    if mapType is None: 
        mapType = mapInfo['type']
    
    if mapType == 0: 
        x = err
    elif mapType == 1: 
        x = 10**(eRange * err + eMin)
    elif mapType == 2: 
        x = 10**((eRange/2 * err) - (eMax - eMin)/2)
    return x

mapInfo = {'type': 0, 
           'eMin': -8, # from 1e-8 to 1e-0 
           'eMax': 0}

# calculate RMSE in given dimension
def torch_rmse(data, dimension=1): 
    return torch.sqrt(torch.mean(torch.square(data), dimension))

# helper function for parametervariation
def GetResLayerList(iResLayerType): 
    return [[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3], [5], [6], [7]][iResLayerType]

#%% 
# utility function to check if the file already exists to skip generation of the file
def checkCreateData(dataFile): 
    createData = True
    myDir = ''
    for myStr in dataFile.split('/')[:-1]: 
        myDir  =  os.path.join(myDir, myStr)
    fileName = dataFile.split('/')[-1]
    try: 
        for files in os.listdir(myDir): 
            if fileName in files: 
                if __name__ == '__main__': # file is loaded by 
                    print('datafile already exists, skip creation!') 
                createData = False
    except: 
        pass
        
    return createData

#%% +++++++++++++++++++++++++++++++++++++++++++++++++++++

# input: 
#    * mbs: the Exudyn multibody system
#    * A_d: the factor to which the initial solution should decay
#    * nValues: number of eigenvalues returned. if nValues = 0 all values are returned. 
# output:
#    * t_d: damping times of coordinates
#    * eVal: smallest eigenvalues. Eigenvalues equal or very close to zeropoint out undamped coordinates. 
def getDamping(mbs, A_d, t = None, flagDebug = True, nValues = 1, computeComplexEigenvalues=True): 
    
    [eigenValues, eVectors] = mbs.ComputeODE2Eigenvalues(computeComplexEigenvalues=computeComplexEigenvalues,
                                                         useAbsoluteValues=False)
    # here the sqrt is already taken and eigenvalues are sorted
    w, D, eVal = [], [], []
    for i in range(len(eigenValues)): 
        if eigenValues[i].imag == 0: 
            pass
        else: 
            eigenFreq = abs(eigenValues[i].imag)
            eigenD = abs(eigenValues[i].real)/abs(eigenValues[i].imag)
            if not(eigenFreq in w): 
                w += [eigenFreq]
                D += [eigenD] 

        eVal += [abs(np.real(eigenValues[i]))]
        if not(computeComplexEigenvalues):
            eVal[-1] = np.sqrt(eVal[-1]) # when not calculating the complex eigenvalues we get the squared eigenvalues... 

    # the eigenvalues are already sorted.
    t_dMax = - np.log(A_d)/(eVal[-nValues::])
    
    if flagDebug: 
        print('largest damping time constant: ', roundSignificant(t_dMax, 3), '\n')
    return t_dMax, eVal[-nValues::] 



#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++
# translates integer values into structures for parametrized tests; e.g. NeuralNetworkStructureTypes()[2] == 'RL'
def NeuralNetworkStructureTypes():
    return ['L', 'LL', 'RL', 'LR', 'SL', 'LS', 'LRL', 'LLL', 'LRLRL', 'LLR',
            'LLRL', 'LLLRL', 'LLRLRL', 'LLLL', 'LRdL', 'LdLRdL', 'LLRdLRdL', 'LLLRLRL', 
            'R', 'RR', 'RLR', 'RRR', 'RLRL', 'LRLR', 'RLRLR', 
            'LRLR', 'LLRLLR', 'LRLRLRLR', 'LLRLLRLLRLLR', '', 'LLRLLRL', 
            ]

# translates type strings into classes: Recurrent Neural Networks, Feedforward 
# neural Networks, Long Short Term Memory and Convolutional Neural Networks
def BasicNeuralNetwork(typeName):
    if typeName == 'RNN':
        return nn.RNN
    elif typeName == 'FFN':
        return nn.Linear
    elif typeName == 'LSTM': #not ready; needs extension for (h0,c0) hidden state
        return nn.LSTM
    elif typeName == 'CNN': 
        return nn.Conv2d
    raise ValueError('BasicNeuralNetwork: invalid type: ', typeName)
    return None


def BasicNeuralNetworkNames():
    return ['RNN', 'FFN', 'LSTM', 'CNN']


variableShortForms = {'maxEpochs':'EP', 'learningRate':'LR', 'nTraining':'NT',
                      'batchSize':'BS', 'hiddenLayerSize':'HL', 'hiddenLayerStructureType':'NN'}

# translates full names to abbreviated names as shown above
def VariableShortForm(var):
    if var in variableShortForms:
        return variableShortForms[var]
    else:
        return var

# currently 12 possible activation types, although this does not change much
def GetActivationType(): 
    activationFuncions = [nn.ReLU, nn.ELU, nn.GELU, nn.GLU, nn.Tanh, nn.CELU, nn.CELU,
                          nn.Hardsigmoid, nn.Hardtanh, nn.SiLU, nn.Mish, nn.LeakyReLU]
    return activationFuncions

# function: creates a serial neuronal network from pytorch modules
# input: NeuralNetworkTypeName can be 'RNN', 'CNN' or 'FNN'. 
#
# output: 
#
# note: 
def CreatePresetNeuralNetwork(NeuralNetworkTypeName, NeuralNetworkStructure, inputSize, outputSize, hiddenLayerSize, numberOfLayers=1, nonLinearity = 'tanh',
                              optionalDict={}, resLayerList = [], useBias=True, activationType = 0): 
    NNtype = BasicNeuralNetwork(NeuralNetworkTypeName)
    
    if NeuralNetworkTypeName  == 'RNN' or NeuralNetworkTypeName == 'LSTM':
        inputSize = inputSize[-1] # only first 
        Network = NNtype(inputSize, hiddenLayerSize, batch_first=True, #batch_first=True means that batch dimension is first one in input/output vectors
                                           num_layers=1,
                                           nonlinearity=nonLinearity)
        model = nn.Sequential(Network)
        
    elif NeuralNetworkTypeName  == 'CNN': 
        Network = NNtype(1,1, 3, padding=(1)) # optionalDict['CNNKernelSize'])
        model = nn.Sequential(Network)
        
        model.append(nn.Flatten(1))
        model.append(nn.Linear(np.prod(inputSize), hiddenLayerSize))
    
    elif NeuralNetworkTypeName  == 'FFN':
        
        flatten = nn.Flatten(1)
        Network = NNtype(np.prod(inputSize), hiddenLayerSize, bias=useBias) # first layer is linear
        # when input and output are time vectors the matrix connecting them should be a
        # lower triangular matrix, otherwise the network would "look into the future"
        # it is possible to initialize the weights accordingly but it does not seem to make a real difference in the training
        # if inputSize[0] > hiddenLayerSize: 
        #     print('initialize Network with lower triangular Matrix')
        #     with torch.no_grad(): 
        #         i_shapes = inputSize[0]-hiddenLayerSize
        #         Network.weight[:,i_shapes:].copy_(torch.tril(Network.weight[:,i_shapes:]))
        # elif 
        

        model = nn.Sequential(flatten, Network) 
        
    for i, c in enumerate(NeuralNetworkStructure):
        if i in resLayerList: 
            print('layer {}: residual layer! '.format(i))
        if c.upper() == 'L':
            # pass
            model.append(nn.Linear(hiddenLayerSize, hiddenLayerSize, bias=useBias))
        elif c.upper() == 'R':
            # model.append(nn.ReLU())
            model.append(GetActivationType()[activationType]())
        elif c.upper() == 'S':
            model.append(nn.Sigmoid())
        elif c == 'D':
            model.append(nn.Dropout(0.5))
        elif c == 'd':
            model.append(nn.Dropout(0.2))
        else:
            raise ValueError('MyNeuralNetwork: invalid layer type: '+c)
            
            

    model.append(nn.Linear(hiddenLayerSize, np.prod(outputSize), bias=useBias))
    if len(NeuralNetworkStructure) == 0: 
        model =  NNtype(np.prod(inputSize), np.prod(outputSize), bias=useBias)
        model = nn.Sequential(flatten, model) 
    if NeuralNetworkTypeName != 'RNN' and np.prod(outputSize) > 1: 
        model.append(nn.Unflatten(-1, outputSize))
    return model




#%%+++++++++++++++++++++++++++
def ExtendResultsFile(resultsFile, infoDict, parameterDict={}):
    with open(resultsFile, "r") as f: 
        contents = f.readlines()

    contentsNew = contents[0:3]
    contentsNew += ['#info:'+str(infoDict).replace(' ','').replace('\n','\\n')+'\n']
    contentsNew += ['#params:'+str(parameterDict).replace(' ','').replace('\n','\\n')+'\n']
    contentsNew += contents[4:]

    with open(resultsFile, "w") as f:
        for line in contentsNew:
            f.write(line)

# %%      
def roundSignificant(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++
#NeuralNetwork helper class
class MyNeuralNetwork(nn.Module):
    #computeDevice =['cpu' | 'cuda']
    def __init__(self, inputOutputSize, 
                 hiddenLayerSize, hiddenLayerStructure,
                 neuralNetworkTypeName = 'RNN', computeDevice = 'cpu', 
                 rnnNumberOfLayers = 1,
                 rnnNonlinearity = 'tanh', #'relu', None; 'tanh' is the default
                 resLayerList  = [], 
                 useBias = True, 
                 activationFunction = 0
                 ):
        super().__init__()
        self.computeDevice = torch.device(computeDevice)
        # print('torch NN device=',computeDevice)
        self.hiddenLayerSize = hiddenLayerSize
        self.hiddenLayerStructure = hiddenLayerStructure
        self.rnnNumberOfLayers = rnnNumberOfLayers
        self.initializeHiddenStates = []
        self.neuralNetworkTypeName = neuralNetworkTypeName
        self.neuralNetworkBase = BasicNeuralNetwork(neuralNetworkTypeName)
        self.rnnNonlinearity = rnnNonlinearity
        

            
        self.inputSize = inputOutputSize[0]
        self.outputSize = inputOutputSize[1]
        inputSize = self.inputSize
        outputSize = self.outputSize

        self.myNN = CreatePresetNeuralNetwork(self.neuralNetworkTypeName, self.hiddenLayerStructure, 
                                             self.inputSize, self.outputSize, self.hiddenLayerSize, resLayerList = [], 
                                             useBias = useBias)
            
    def SetInitialHiddenStates(self, hiddenStatesBatch):
        if self.neuralNetworkTypeName == 'RNN':
            self.initializeHiddenStates = hiddenStatesBatch


    def forward(self, x):
        if self.neuralNetworkTypeName == 'RNN' or self.neuralNetworkTypeName == 'LSTM':
            # print('x=', x.size())
            batchSize = x.size(0)
            hidden = self.initialize_hidden_state(batchSize).to(self.computeDevice)
            #hidden = self.initialize_hidden_state(batchSize).to(x.get_device())
            # self.myNN(x)
            out, _ = self.myNN[0](x, hidden)
            out = out.contiguous().view(-1, self.hiddenLayerSize)
            out = self.myNN[1:](out)
            return out #self.myNN(x)
            
        elif self.neuralNetworkTypeName == 'FFN':
            # x = self.NNFLatten(x)
            # out = self.myNN(x)
            # out = x
            # for layer in self.myNN: 
            #     out = layer(out)
            # return out
            return self.myNN(x)
        
        elif self.neuralNetworkTypeName == 'CNN':
            # myInp = torch.rand(1,1,40,45,)
            # batch size x input dimension x len_y, len_x
            x  = x.view(-1,1,x.shape[-2],x.shape[-1]) 
            # -1 for batch size; 
            # x.view(-1,)
            out = self.myNN(x)
            # self.rnn = 
            # print(out.shape)
            return out 
        
            # if 0: 
            #     fc = nn.Linear(6*40, 2*40)
                
                
                
        for i, item in enumerate(self.hiddenLayersList):
            # if i > 1: 
            #     out2 = nn.MaxPool2d((2,1))(out)
            out = item(out)
            # print(out.shape)
            
        #out = self.lastLayer(out)
        # if self.neuralNetworkTypeName == 'FFN': 
            # out = self.UnFlatten(out)
        
        return out


    def initialize_hidden_state(self, batchSize):
        hs = torch.zeros((self.rnnNumberOfLayers, batchSize, self.hiddenLayerSize)).to(self.computeDevice)
        if self.neuralNetworkTypeName == 'RNN' and self.initializeHiddenStates.size(1) != 0:
            nInitAvailable = self.initializeHiddenStates.size(1)
            hs[0,:,:nInitAvailable] = self.initializeHiddenStates

        # for i, hsInit in enumerate(self.initializeHiddenStates):
        #     hs[0,:,i] = hsInit
        return hs

moduleNntc = None #must be set in calling module for multiprocessing

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++
#parameter function for training with exudyn ParameterVariation(...)
def ParameterFunctionTraining(parameterSet):
    global moduleNntc
    
    #++++++++++++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++++++++++++
    #store default parameters in structure (all these parameters can be varied!)
    class P: pass #create emtpy structure for parameters; simplifies way to update parameters

    #default values
    P.maxEpochs=1000 #sufficient
    P.learningRate= 0.001
    P.lossThreshold=1e-8
    P.batchSize=64
    P.neuralNetworkType = 1 #[0=RNN, 1=FFN]; note that the RNN is currently not functioning
    P.hiddenLayerSize=128
    P.hiddenLayerStructureType=0 #'L'
    P.nTraining = 512
    P.nTest = 20
    P.testEvaluationInterval = 0 #number of epochs after which tests are evaluated; 0=no evaluation
    P.lossLogInterval = 20
    P.epochPrintInterval = 100
    P.computationIndex = None
    P.storeModelName = ''
    P.dataFile = None
    P.rnnNonlinearity = 'tanh'
    P.dataLoaderShuffle = False
    P.case = 0 #
    P.computeDevice = 'cpu'
    P.customNetwork = None
    P.customNetworkEst = None
    P.resLayerList = []
    P.typeNumber = 0
    P.useErrorEstimator = False
    P.maxEpochsEst = -1
    P.hiddenLayerSizeEst = -1
    P.useBias = True
    P.estOnOutput = False
    P.activationFunction = 0
    P.verboseMode = 0
    P.mapErrorInfo = {}
    P.outputScalingEst = None
    
    # #now update parameters with parameterSet (will work with any parameters in structure P)
    for key,value in parameterSet.items():
        setattr(P,key,value)
        if 'resLayerType' in key:  
            P.resLayerList = GetResLayerList(value)
            
    #functionData are some values that are not parameter-varied but may be changed for different runs!
    if 'functionData' in parameterSet:
        for key,value in P.functionData.items():
            if key in parameterSet:
                print('ERROR: duplication of parameters: "'+key+'" is set BOTH in functionData AND in parameterSet and would be overwritten by functionData; Computation will be stopped')
                raise ValueError('duplication of parameters')
            setattr(P,key,value)
            if 'resLayerType' in key: 
                P.resLayerList = GetResLayerList(value)
                
    if 'cutInput' in parameterSet['functionData'].keys() and 'nOut' in parameterSet['functionData'].keys(): 
        flagTestCuttingData = True
        nOut = parameterSet['functionData']['nOut']
        nCut = parameterSet['functionData']['cutInput']
        P.inputOutputSizeNN[0] =  (P.inputOutputSizeNN[0][0] - nCut, 1)
        P.inputOutputSizeNN[1] = (nOut, 1)
    else: 
        flagTestCuttingData = False
        
    
    print('compute device: ', P.computeDevice)
    if P.maxEpochsEst == -1: 
        P.maxEpochsEst = P.maxEpochs
    if P.hiddenLayerSizeEst == -1: 
        P.hiddenLayerSizeEst = P.hiddenLayerSize
    hiddenLayerStructure = NeuralNetworkStructureTypes()[int(P.hiddenLayerStructureType)] #'L'
    neuralNetworkTypeName = BasicNeuralNetworkNames()[P.neuralNetworkType]
    
    #print(neuralNetworkTypeName )
    #++++++++++++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++++++++++++
    if P.customNetwork is None: 
        myNNModel = MyNeuralNetwork(inputOutputSize =      P.inputOutputSizeNN,# input and output size, 
                                   neuralNetworkTypeName = neuralNetworkTypeName,
                                   hiddenLayerSize =       P.hiddenLayerSize,
                                   hiddenLayerStructure =  hiddenLayerStructure,
                                   rnnNonlinearity =       P.rnnNonlinearity,
                                   computeDevice =         P.computeDevice, 
                                   resLayerList  =         P.resLayerList, 
                                   useBias  =              P.useBias, 
                                   activationFunction =    P.activationFunction, 
                                   )
        if P.verboseMode > 0: 
            print("*"*10 + "\n NN Structure: " + hiddenLayerStructure + '\n' + "*"*10)
    else: 
        myNNModel = P.customNetwork(inputOutputSize =      P.inputOutputSizeNN,# input and output size, 
                                   neuralNetworkTypeName = neuralNetworkTypeName,
                                   hiddenLayerSize =       P.hiddenLayerSize,
                                   hiddenLayerStructure =  hiddenLayerStructure,
                                   rnnNonlinearity =       P.rnnNonlinearity,
                                   computeDevice=          P.computeDevice, 
                                   resLayerList  =         P.resLayerList, 
                                   typeNumber   =          P.typeNumber, 
                                   )
        print('custom network: \n{}'.format(myNNModel))
    nnModelEst = None
    if P.useErrorEstimator: 
        if P.estOnOutput: 
            sizeEst = (P.inputOutputSizeNN[1], 1)
        else: 
            sizeEst = (P.inputOutputSizeNN[0], 1)
        if P.customNetworkEst is None: 
            nnModelEst = MyNeuralNetwork(inputOutputSize =     sizeEst,# input and output size, 
                                       neuralNetworkTypeName = neuralNetworkTypeName, #+ str('_ErrorEstimator'),
                                       hiddenLayerSize =       P.hiddenLayerSizeEst,
                                       hiddenLayerStructure =  hiddenLayerStructure,
                                       rnnNonlinearity =       P.rnnNonlinearity,
                                       computeDevice =         P.computeDevice, 
                                       useBias  =              P.useBias
                                       # resLayerList  =         P.resLayerList, 
                                       )
        else: 
            nnModelEst = P.customNetworkEst(inputOutputSize =     sizeEst,# input and output size, 
                                       neuralNetworkTypeName = neuralNetworkTypeName, #+ str('_ErrorEstimator'),
                                       hiddenLayerSize =       P.hiddenLayerSizeEst,
                                       hiddenLayerStructure =  hiddenLayerStructure,
                                       rnnNonlinearity =       P.rnnNonlinearity,
                                       computeDevice =         P.computeDevice, 
                                       resLayerList  =         P.resLayerList, 
                                       typeNumber   =          P.typeNumber, 
                                       # resLayerList  =         P.resLayerList, 
                                       )
        nnModelEst.information = 'ErrorEstimator'
        nnModelEst.mapErrorInfo = P.mapErrorInfo # map error in log scale
        nnModelEst.maxEpochs = P.maxEpochs
        nnModelEst.outputScaling = P.outputScalingEst
        
    
    nntc = NeuralNetworkTrainingCenter(simulationModel=None, 
                                       nnModel = myNNModel, 
                                       computeDevice=P.computeDevice, 
                                       inputOutputSizeNN = P.inputOutputSizeNN, 
                                       nnModelEst = nnModelEst) 
    
    if P.dataFile == None:
        moduleNntc.CreateTrainingAndTestData(nTraining=P.nTraining, nTest=P.nTest,
                                       #parameterFunction=PVCreateData, #for multiprocessing
                                       )
    else:
        moduleNntc.LoadTrainingAndTestsData(P.dataFile, nTraining=P.nTraining, nTest=P.nTest)   # for evaluation
        nntc.LoadTrainingAndTestsData(P.dataFile, nTraining=P.nTraining, nTest=P.nTest)         # for training
        # nntc.inp
    
            


    # # this flag is only fot testing 
    if flagTestCuttingData:        
        nntc.inputsTraining = nntc.inputsTraining[:,nCut:,:]
        nntc.inputsTest = nntc.inputsTest[:,nCut:,:]
        
        nIn = nntc.inputsTraining.shape[1]
        nntc.targetsTraining = nntc.targetsTraining[:,(nIn-nOut):,:]
        nntc.targetsTest = nntc.targetsTest[:,(nIn-nOut):,:]
        
        if P.useErrorEstimator: 
            nntc.inputsTestEst = nntc.inputsTestEst[:,nCut:,:]
            nntc.inputsTrainingEst = nntc.inputsTrainingEst[:,nCut:,:]
            nntc.targetsTestEst = nntc.targetsTestEst[:,(nIn-nOut):,:]
            nntc.targetsTrainingEst = nntc.targetsTrainingEst[:,(nIn-nOut):,:]
        
        
    
    # if 1: 
        # return myNNModel
    nntc.TrainModel(maxEpochs =              P.maxEpochs, 
                    learningRate =           P.learningRate, 
                    lossThreshold =          P.lossThreshold, 
                    batchSize =              P.batchSize,
                    # hiddenLayerSize =        P.hiddenLayerSize, 
                    # hiddenLayerStructure =   hiddenLayerStructure,
                    testEvaluationInterval = P.testEvaluationInterval,
                    lossLogInterval =        P.lossLogInterval,
                    dataLoaderShuffle =      P.dataLoaderShuffle,
                    seed =                   P.case,
                    maxEpochsEst    =        P.maxEpochsEst, 
                    epochPrintInterval =     P.epochPrintInterval)

    rv = nntc.EvaluateModel()
    rv['lossEpoch'] = list(nntc.LossLog()[:,0])
    rv['loss'] = list(nntc.LossLog()[:,1])

    rv['testResultsEpoch'] = nntc.validationResults[0]
    rv['testResults'] = nntc.validationResults[1]
    rv['testResultsMin'] = nntc.validationResults[2]
    rv['testResultsMean'] = nntc.validationResults[3]
    rv['testResultsMax'] = nntc.validationResults[4]
    
    if nntc.useErrorEstimator: 
        rv['lossEpochEst'] = list(np.array(nntc.lossLogEst)[:,0])
        rv['lossEst'] = list(np.array(nntc.lossLogEst)[:,1])
        rv['valEpochEst'] = list(np.array(nntc.lossValErrEst)[:,0])
        rv['velErr'] = list(np.array(nntc.lossValErrEst)[:,1])
        
        
    if P.storeModelName!='':
        nntc.SaveNNModel(P.storeModelName+str(P.computationIndex)+'.pth')

    if P.useErrorEstimator: 
        return rv, myNNModel, nnModelEst
    else: 
        return rv, myNNModel,  #dictionary


# helper function to count how many runs are neccessary to process the list opf parameters
def GetNumOfParameterRuns(parameters): 
    numRuns = 1
    for val in parameters.values(): 
        numRuns *= len(val)
    return numRuns

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++
#create training and test data
#parameter function, allowing to run parallel using the global moduleNntc object
def PVCreateData(parameterFunction):
    global moduleNntc
    
    simModel = moduleNntc.GetSimModel()
    isTest = parameterFunction['functionData']['isTest']
    nSamples = parameterFunction['functionData']['nSamples']
    cnt = parameterFunction['cnt'] #usually not needed
    dataErrorEstimator = parameterFunction['functionData']['dataErrorEstimator']
    
    flattenData = False # parameterFunction['functionData']['flattenData']
    if cnt % 100 == 0: 
        print('cnt: ', cnt, flush=True)
    seed = int(cnt)
    if isTest:
        seed = 2**31 - seed #2**32-1 is max value
        #print('seed:', seed)
    np.random.seed(seed)

    inputVec = simModel.CreateInputVector(relCnt = cnt/nSamples, isTest = isTest, dataErrorEstimator=dataErrorEstimator)
    hiddenVec = simModel.CreateHiddenInit(isTest)
    outputVec = simModel.ComputeModel(inputVec, hiddenVec)

    if flattenData:
        return [[inputVec.flatten()], outputVec.flatten(), hiddenVec.flatten()]
    else:
        return [inputVec, outputVec, hiddenVec]

    # inputs += [[inputVec.flatten()]]
    # targets += [outputVec.flatten()]


class NeuralNetworkTrainingCenter():
    #create class with model to be trained
    #createModelDict is used to initialize CreateModel function with specific parameters
    #initialize seed
    def __init__(self, nnModel, simulationModel=SimulationModel(), createModelDict={}, inputOutputSizeNN = None, 
                 computeDevice = 'cpu', verboseMode=1, nnModelEst = None):
        
        self.nnModel = nnModel
        self.simulationModel = simulationModel
        self.verboseMode = verboseMode
        

        # initialize for pure loading of model:
        self.computeDevice = computeDevice
        self.lossFunction = nn.MSELoss() #mean square error, equals: np.linalg.norm(x-y)**2/len(x)
        self.useErrorEstimator = not(nnModelEst is None)
        self.nnModelEst = nnModelEst
        self.lossFunctionEst = nn.MSELoss() # 

        self.inputsTraining = []
        self.targetsTraining = []
        self.inputsTest = []
        self.targetsTest = []

        #initialization of hidden layers in RNN (optional)
        self.hiddenInitTraining = []
        self.hiddenInitTest = []

        self.floatType = torch.float #16, 32, 64
        try: 
            self.simulationModel.CreateModel(**createModelDict)
        except: 
            self.inputOutputSizeNN = inputOutputSizeNN
            # print('warning: model could not be created...')
            # self.simulationModel = None
    
    #get stored NN model
    def GetNNModel(self):
        return self.nnModel
    
    # set stored NN model
    def SetNNModel(self, newNeuralNetwork): 
        self.nnModel = newNeuralNetwork
    
    
    # get stored simulation model
    def GetSimModel(self):
        return self.simulationModel
    
    
    

    #serial version to create data
    def CreateData(self,parameterFunction):
        simulationModel = self.GetSimModel()
        isTest = parameterFunction['functionData']['isTest']
        nSamples = parameterFunction['functionData']['nSamples']
        showTests = parameterFunction['functionData']['showTests']
        dataErrorEstimator = parameterFunction['functionData']['dataErrorEstimator']
        
        cnt = parameterFunction['cnt'] #usually not needed
        # flattenData = False # parameterFunction['functionData']['flattenData']
        
        inputVec = simulationModel.CreateInputVector(relCnt = cnt/nSamples, isTest = isTest, 
                                                     dataErrorEstimator = dataErrorEstimator, )
        # if dataErrorEstimator: 
        #     inputVec *= 3
        hiddenVec = simulationModel.CreateHiddenInit(isTest)
        outputVec = simulationModel.ComputeModel(inputVec,
                                         hiddenData=hiddenVec, 
                                         solutionViewer = isTest and (cnt in showTests))
    
        # if flattenData:
            # return [[inputVec.flatten()], outputVec.flatten(), hiddenVec.flatten()]
        # else:
            # print('inputs shape=', inputVec.shape)
        if hasattr(simulationModel, 'nCutInputSteps'):
            inputVec = inputVec            
        return [inputVec, outputVec, hiddenVec]
    
    #create input data from model function
    #if plotData > 0, the first data is plotted
    #showTests is a list of tests which are shown with solution viewer (not parallel)
    def CreateTrainingAndTestData(self, nTraining, nTest, 
                                  parameterFunction=None, showTests=[], numberOfThreads=None):
        
        useMultiProcessing=True
        if parameterFunction is None:
            parameterFunction = self.CreateData
            useMultiProcessing=False
        else:
            parameterFunction = PVCreateData
            if showTests != []:
                print('CreateTrainingAndTestData: showTests does not work with multiprocessing')
            
        self.inputsTraining = []
        self.targetsTraining = []
        self.inputsTest = []
        self.targetsTest = []


        self.inputsTrainingEst = []
        self.targetsTrainingEst = []
        self.inputsTestEst = []
        self.targetsTestEst = []
        
        

        #initialization of hidden layers in RNN (optional)
        self.hiddenInitTraining = []
        self.hiddenInitTest = []
        
        self.hiddenInitTrainingEst = []
        self.hiddenInitTestEst = []
    

        self.nTraining = nTraining
        self.nTest = nTest
        
        try:
            nThreads = os.environ['SLURM_NTASKS']
            flagMPI = True
        except:
            nThreads = os.cpu_count()
            flagMPI = False
        if useMultiProcessing:  
            print('create data by multiprocessing using {} threads'.format(nThreads), ', using MPI'*flagMPI, flush=True)


        # flattenData = self.simulationModel.IsFFN()
        if self.useErrorEstimator: 
            modes = ['training', 'test', 'trainingEstimator', 'testEstimator']
        else: 
            modes = ['training', 'test']
        for mode, modeStr in enumerate(modes):
            if self.verboseMode>0:
                print('create '+modeStr+' data ...')
            nData = nTraining if mode==0 or mode==2 else nTest
            if 'Estimator' in modeStr: 
                dataErrorEstimator = True
            else: 
                dataErrorEstimator = False 
                
            paramDict = {}
            if numberOfThreads != None:
                paramDict['numberOfThreads'] = numberOfThreads
            else: 
                paramDict['numberOfThreads'] = nThreads 
                
            #+++++++++++++++++++++++++++++++++++++++++++++
            #create training data
            
            [parameterDict, values] = ParameterVariation(parameterFunction, parameters={'cnt':(0,nData-1,nData)},
                                                         useMultiProcessing=useMultiProcessing and nData>1,
                                                         showProgress=self.verboseMode>0, **paramDict,
                                                         useMPI=flagMPI, 
                                                         parameterFunctionData={'isTest':mode==1 or mode == 3, 
                                                                                'nSamples':nData,
                                                                                'showTests':showTests,
                                                                                'dataErrorEstimator': dataErrorEstimator})
                                                                                # 'flattenData':flattenData})
    
            for i, item in enumerate(values):
                if type(item[0]) is tuple: 
                    item[0] = item[0][0]
                    if i == 0: 
                        print('Warning: Model should be 6R Flexible Robot, therefore the input is modified to cut off the trajectory object!')
                if mode == 0:
                    self.inputsTraining += [item[0]]
                    self.targetsTraining += [item[1]]
                    self.hiddenInitTraining += [item[2]]
                elif mode == 1:
                    self.inputsTest += [item[0]]
                    self.targetsTest += [item[1]]
                    self.hiddenInitTest += [item[2]]
                    
                elif mode == 2: 
                    self.inputsTrainingEst += [item[0]]
                    self.targetsTrainingEst += [item[1]]
                    self.hiddenInitTrainingEst += [item[2]]
                    
                elif mode == 3: 
                    self.inputsTestEst += [item[0]]
                    self.targetsTestEst += [item[1]]
                    self.hiddenInitTestEst += [item[2]]
                    
                    
                else: 
                    raise ValueError 
                        
        # stack for loading data in batches
        
        #convert such that torch does not complain about initialization with lists:
        self.inputsTraining = np.stack(self.inputsTraining, axis=0) # this is the same as np.array(self.inputsTraining)
        self.targetsTraining = np.stack(self.targetsTraining, axis=0)
        self.inputsTest = np.stack(self.inputsTest, axis=0)
        self.targetsTest = np.stack(self.targetsTest, axis=0)

        self.hiddenInitTraining = np.stack(self.hiddenInitTraining, axis=0)
        self.hiddenInitTest = np.stack(self.hiddenInitTest, axis=0)

        # cut initial acceleration scheme
        if hasattr(self.GetSimModel(), 'nCutInputSteps'): 
            nStepsCutInput = self.GetSimModel().nCutInputSteps
            self.inputsTraining = self.inputsTraining[:,:,nStepsCutInput:]
            self.inputsTest = self.inputsTest[:,:,nStepsCutInput:]
            
        if self.useErrorEstimator: 
            self.inputsTrainingEst = np.stack(self.inputsTrainingEst, axis=0) # this is the same as np.array(self.inputsTraining)
            self.targetsTrainingEst = np.stack(self.targetsTrainingEst, axis=0)
            self.inputsTestEst = np.stack(self.inputsTestEst, axis=0)
            self.targetsTestEst = np.stack(self.targetsTestEst, axis=0)

            self.hiddenInitTrainingEst = np.stack(self.hiddenInitTrainingEst, axis=0)
            self.hiddenInitTestEst = np.stack(self.hiddenInitTestEst, axis=0)

            # cut initial acceleration scheme from error estimator
            if hasattr(self.GetSimModel(), 'nCutInputSteps'): 
                self.inputsTrainingEst = self.inputsTrainingEst[:,:,nStepsCutInput:]
                self.inputsTestEst = self.inputsTestEst[:,:,nStepsCutInput:]
        
        
            

    #save data to .npy file
    def SaveTrainingAndTestsData(self, fileName):
        fileExtension = ''
        if len(fileName) < 4 or fileName[-4:]!='.npy':
            fileExtension = '.npy'
        
        os.makedirs(os.path.dirname(fileName+fileExtension), exist_ok=True)
        
        dataDict = {}
        
        dataDict['version'] = 2 #to check correct version
        dataDict['ModelName'] = self.GetSimModel().GetModelName() #to check if correct name
        dataDict['inputShape'] = self.GetSimModel().GetInputScaling().shape #to check if correct size
        dataDict['outputShape'] = self.GetSimModel().GetOutputScaling().shape #to check if correct size
        dataDict['nTraining'] = self.nTraining
        dataDict['nTest'] = self.nTest
        dataDict['inputsTraining'] = self.inputsTraining
        dataDict['targetsTraining'] = self.targetsTraining
        dataDict['inputsTest'] = self.inputsTest
        dataDict['targetsTest'] = self.targetsTest

        #initialization of hidden layers in RNN (optional)
        dataDict['hiddenInitTraining'] = self.hiddenInitTraining
        dataDict['hiddenInitTest'] = self.hiddenInitTest

        dataDict['floatType'] = self.floatType
        
        #version 2 from here
        if self.useErrorEstimator: 
            dataDict['inputsTrainingEst'] = self.inputsTrainingEst
            dataDict['targetsTrainingEst'] = self.targetsTrainingEst
            dataDict['inputsTestEst'] = self.inputsTestEst
            dataDict['targetsTestEst'] = self.targetsTestEst
            
            dataDict['hiddenInitTrainingEst'] = self.hiddenInitTrainingEst
            dataDict['hiddenInitTestEst'] = self.hiddenInitTestEst
        
        with open(fileName+fileExtension, 'wb') as f:
            np.save(f, dataDict, allow_pickle=True) #allow_pickle=True for lists or dictionaries
    
    #load data from .npy file
    #allows loading data with less training or test sets
    def LoadTrainingAndTestsData(self, fileName, nTraining=None, nTest=None):
        fileExtension = ''
        if len(fileName) < 4 or fileName[-4:]!='.npy':
            fileExtension = '.npy'
            
        with open(fileName+fileExtension, 'rb') as f:
            dataDict = np.load(f, allow_pickle=True).all()   #allow_pickle=True for lists or dictionaries; .all() for dictionaries
            
        if dataDict['version'] >= 1:
            # in newer versions data shape is contained in the neural network (flatten / unlatten) instead of the data
            # therefore the data does not need to be changed according to model
            if dataDict['version'] < 2: 
                if dataDict['modelName'] != self.GetNNModel().GetModelName(): #to check if correct name
                    raise ValueError('NeuralNetworkTrainingCenter.LoadTrainingAndTestsData: model name does not match current model')
                if dataDict['inputShape'] != self.GetNNModel().GetInputScaling().shape:
                    raise ValueError('NeuralNetworkTrainingCenter.LoadTrainingAndTestsData: inputShape does match current model inputSize')
                if dataDict['outputShape'] != self.GetNNModel().GetOutputScaling().shape:
                    raise ValueError('NeuralNetworkTrainingCenter.LoadTrainingAndTestsData: outputShape does match current model outputShape')

            if nTraining == None:
                nTraining = dataDict['nTraining']
            if nTest == None:
                nTest = dataDict['nTest']
            
            if dataDict['nTraining'] < nTraining:
                raise ValueError('NeuralNetworkTrainingCenter.LoadTrainingAndTestsData: available training sets ('+
                                 str(dataDict['nTraining'])+') are less than requested: '+str(nTraining))
            if dataDict['nTest'] < nTest:
                raise ValueError('NeuralNetworkTrainingCenter.LoadTrainingAndTestsData: available test sets ('+
                                 str(dataDict['nTest'])+') are less than requested: '+str(nTest))

            self.nTraining = nTraining
            self.nTest = nTest
            self.inputsTraining = dataDict['inputsTraining'][:nTraining]
            self.targetsTraining = dataDict['targetsTraining'][:nTraining]
            self.hiddenInitTraining = dataDict['hiddenInitTraining'][:nTraining]
            self.inputsTest = dataDict['inputsTest'][:nTest]
            self.targetsTest = dataDict['targetsTest'][:nTest]
            self.hiddenInitTest = dataDict['hiddenInitTest'][:nTest]

            self.floatType = torch.float
            if self.useErrorEstimator:
                self.floatType = dataDict['floatType']
            
            if self.useErrorEstimator: 
                if not('inputsTrainingEst' in dataDict.keys()): 
                    raise ValueError('LoadTrainingAndTestData failed because of missing data for Estimator. Is the data created correctly?')
                self.inputsTrainingEst = dataDict['inputsTrainingEst'][:nTraining]
                self.targetsTrainingEst = dataDict['targetsTrainingEst'][:nTraining]
                self.hiddenInitTrainingEst = dataDict['hiddenInitTrainingEst'][:nTraining]
                self.inputsTestEst = dataDict['inputsTestEst'][:nTest]
                self.targetsTestEst = dataDict['targetsTestEst'][:nTest]
                self.hiddenInitTestEst = dataDict['hiddenInitTestEst'][:nTest]
                
        #convert such that torch does not complain about initialization with lists:
        self.inputsTraining = np.stack(self.inputsTraining, axis=0)
        self.targetsTraining = np.stack(self.targetsTraining, axis=0)
        self.inputsTest = np.stack(self.inputsTest, axis=0)
        self.targetsTest = np.stack(self.targetsTest, axis=0)
        self.hiddenInitTraining = np.stack(self.hiddenInitTraining, axis=0)
        self.hiddenInitTest = np.stack(self.hiddenInitTest, axis=0)


    def printLoss(self, currentLoss, epoch, maxEpochs, epochPrintInterval, tStart): 
            #printing
            if (epoch+1) % epochPrintInterval == 0:
                tNow = time.time()
                lossStr = "{:.3e}".format(currentLoss)
                if self.verboseMode > 0:
                    print(f'Epoch {epoch+1}/{maxEpochs}, Loss: {lossStr}',end='')
                    print('; t:',round(tNow-tStart,1),'/',round((tNow-tStart)/epoch*maxEpochs,0))

    # check the estimator model by forward pass of input trainingsset
    def GetEstType(self): 
        with torch.no_grad():
            try: 
                self.nnModelEst(torch.tensor(self.inputsTrainingEst, dtype=self.floatType).to(self.computeDevice)[0:2])
                self.estOnOutput = False
            except: 
                self.estOnOutput = True    
    
    
    #create training data
    def TrainModel(self, maxEpochs = 1000, lossThreshold=1e-7,
                   learningRate = 0.001, batchSize = 32,
                   testEvaluationInterval = 0,
                   seed = 0,
                   lossLogInterval = 50,
                   epochPrintInterval = 100,
                   dataLoaderShuffle = False,
                   batchSizeIncrease = 1,           #can be used to increase batch size after initial slower training
                   batchSizeIncreaseLoss = None,    #loss threshold, at which we switch to (larger) batch size
                   # reloadTrainedModel = '',         #load (pre-)trained model after setting up optimizer
                   maxEpochsEst = 1000, 
                   ):

        #%%++++++++++++++++++++++++++++++++++++++++
        self.maxEpochs = int(maxEpochs)
        self.maxEpochsEst = int(maxEpochsEst)
        self.lossThreshold = lossThreshold
        self.learningRate = learningRate
        self.batchSize = int(batchSize)

        self.testEvaluationInterval = int(testEvaluationInterval)
        self.dataLoaderShuffle = dataLoaderShuffle

        self.lossLogInterval = int(lossLogInterval)
        self.seed = int(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(self.computeDevice != 'cuda')
        np.random.seed(seed)
        random.seed(seed)
        
        self.floatType = torch.float # 16, 32, 64
        
        
        #%%++++++++++++++++++++++++++++++++++++++++
        if self.simulationModel != None: 
            [inputSize, outputSize] = self.simulationModel.GetInputOutputSizeNN()
        else: 
            # no simulation model
            inputSize, outputSize = self.inputOutputSizeNN
            nInputs = np.prod(inputSize)
            nOutputs = np.prod(outputSize)
        # self.rnn = self.nnModel
        
        if self.floatType == torch.float64:
            self.nnModel = self.nnModel.double()
        if self.floatType == torch.float16:
            self.nnModel = self.nnModel.half() #does not work in linear/friction
        
        self.nnModel = self.nnModel.to(self.computeDevice)
        if not(self.nnModelEst is None): 
            self.nnModelEst = self.nnModelEst.to(self.computeDevice)
        #adjust for FFN
        # self.AdjustInputsToNN(self.nnModel.neuralNetworkTypeName)
        
        
        
        # Convert your data to PyTorch tensors and create a DataLoader
        inputs = torch.tensor(self.inputsTraining, dtype=self.floatType).to(self.computeDevice)#,non_blocking=True)
        targets = torch.tensor(self.targetsTraining, dtype=self.floatType).to(self.computeDevice)#,non_blocking=True)
        hiddenInit = torch.tensor(self.hiddenInitTraining, dtype=self.floatType).to(self.computeDevice)#,non_blocking=True)
        
        inputsTest = torch.tensor(self.inputsTest, dtype=self.floatType).to(self.computeDevice)
        targetsTest = torch.tensor(self.targetsTest, dtype=self.floatType).to(self.computeDevice)
        hiddenInitTest = torch.tensor(self.hiddenInitTest, dtype=self.floatType).to(self.computeDevice)
        

        dataset = TensorDataset(inputs, targets, hiddenInit)
        dataloader = DataLoader(dataset, batch_size=batchSize,
                                shuffle=self.dataLoaderShuffle)
        # print('dataset on: ', dataset.tensors[0].device)
        
        if batchSizeIncrease != 1:
            dataloader2 = DataLoader(dataset, batch_size=batchSize*batchSizeIncrease,
                                    shuffle=self.dataLoaderShuffle)

        datasetTest = TensorDataset(inputsTest, targetsTest, hiddenInitTest)
        dataloaderTest = DataLoader(datasetTest, batch_size=batchSize, shuffle=self.dataLoaderShuffle)

        
        # Define a loss function and an optimizer
        optimizer = torch.optim.Adam(self.nnModel.parameters(), lr=self.learningRate) # ,  betas=(0.9, 0.999), weight_decay = 1e-6)

        self.lossLog = []
        self.minLoss, self.epochBestLoss, self.nnModelBest = np.inf, -1, None
        self.validationResults = [[], [], [], [], []] #epoch, tests, min, mean, max
        
        # 
        # if reloadTrainedModel!='': #this may cause problems with cuda!
        #     #self.rnn.rnn.load_state_dict(torch.load(reloadTrainedModelDict)) #this does not work
        #     self.LoadNNModel(reloadTrainedModel)
        #     self.rnn = self.rnn.to(self.computeDevice)

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++ MAIN TRAINING LOOP ++++++++++++++++++++++++++++
        tStart = time.time()
        for epoch in range(self.maxEpochs):  
            for inputs, targets, initial_hidden_states in dataloader:
                # Forward pass
                self.nnModel.SetInitialHiddenStates(initial_hidden_states)
                outputs = self.nnModel(inputs)               
                loss = self.lossFunction(outputs, targets)
        
                # Backward pass and optimization
                optimizer.zero_grad() #(set_to_none=True)
                loss.backward()
                optimizer.step()

            currentLoss = loss.item()
            # self.lossList += [currentLoss]
            #switch to other dataloader:
            if batchSizeIncreaseLoss != None and currentLoss < batchSizeIncreaseLoss:
                dataloader = dataloader2

            #log loss at interval
            if (epoch % self.lossLogInterval == 0) or (epoch == self.maxEpochs-1):
                self.lossLog += [[epoch, currentLoss]]
            
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #perform tests after interval
            if testEvaluationInterval > 0 and (epoch > 0 
                                               and epoch%testEvaluationInterval == 0
                                               or epoch == self.maxEpochs-1
                                               or currentLoss < self.lossThreshold):
                self.nnModel.eval() #switch network to evaluation mode
                results = []
                with torch.no_grad(): #avoid calculation of gradients:
                    for inputs, targets, initial_hidden_states in dataloaderTest:
                        # Forward pass
                        self.nnModel.SetInitialHiddenStates(initial_hidden_states)
                        outputs = self.nnModel(inputs)
                        # outputs = outputs.view(-1, *outputSize)  # Reshape the outputs to match the target shape
                        mse = self.lossFunction(outputs, targets)
                        
                        #print(outputs.view(-1, *outputSize))

                        # for i in range(self.nTest):
    
                        #     inputVec = self.inputsTest[i:i+1]
                            
                        #     x = torch.tensor(inputVec, dtype=self.floatType).to(self.computeDevice)
                        #     y = np.array(self.nnModel(x).to(self.computeDevice).tolist()[0]) #convert output to list
                        #     yRef = self.targetsTest[i:i+1][0]
                            
                        #     print('x.size=',x.size())
                        #     print('y.size=',y.shape)
                        #     print('yRef.size=',yRef.shape)
    
                        #     #this operation fully runs on CPU:
                        #     mse = self.lossFunction(torch.tensor(y, dtype=self.floatType), 
                        #                             torch.tensor(yRef, dtype=self.floatType))
                        
                        # results += [float(mse)] #np.linalg.norm(y-yRef)**2/len(y)]
                        # results += torch.mean((outputs - targets)**2,1)
                        results += torch.mean((outputs - targets)**2,1).cpu().numpy().reshape(outputs.shape[0],outputSize[1]).tolist()
                    
                self.validationResults[0] += [epoch]
                self.validationResults[1] += [results]
                self.validationResults[2] += [min(results)]
                self.validationResults[3] += [np.mean(results)]
                self.validationResults[4] += [max(results)]
                # check if current model is better than previous best model
                # if so the model is copied
                if self.validationResults[3][-1] < self.minLoss:
                    self.minLoss = self.validationResults[3][-1]
                    self.nnModelBest = copy.deepcopy(self.nnModel)
                    self.epochBestLoss = epoch
                    print('best minimal (validation) loss: {} at epoch {}'.format(roundSignificant(self.minLoss, 4), self.epochBestLoss))
                    
                #switch back to training!
                self.nnModel.train() # set network into training mode
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if self.verboseMode > 0: 
                self.printLoss(currentLoss, epoch, self.maxEpochs, epochPrintInterval, tStart)
            if currentLoss < self.lossThreshold:
                if self.verboseMode > 0:
                    print('iteration stopped at', epoch,'due to tolerance; Loss:', currentLoss)
                break
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.epochBestLoss != epoch and self.epochBestLoss != -1: 
            self.nnModel = copy.deepcopy(self.nnModelBest)
            print('using currently best model from validation at epoch ', self.epochBestLoss)
            
            
        self.trainingTime = time.time()-tStart 
        if self.verboseMode > 0:
            print('training time=',self.trainingTime)
        
        #%% ---------------------------- error estimator ---------------------------- %%#
        tStart2 = time.time()
        if self.useErrorEstimator: 
            self.nnModel.eval()
            
            self.lossLogEst = []
            self.lossValErrEst = []
            outputScaling = torch.tensor(self.nnModelEst.outputScaling, dtype=self.floatType).to(self.computeDevice)
            # Convert your data to PyTorch tensors and create a DataLoader
            if False: # only use estimator data to train estimator
                inputsEst = torch.tensor(self.inputsTrainingEst, dtype=self.floatType).to(self.computeDevice)#,non_blocking=True)
                targetsEst = torch.tensor(self.targetsTrainingEst, dtype=self.floatType).to(self.computeDevice)#,non_blocking=True)
                
                
                inputsTestEst = torch.tensor(self.inputsTestEst, dtype=self.floatType).to(self.computeDevice)
                targetsTestEst = torch.tensor(self.targetsTestEst, dtype=self.floatType).to(self.computeDevice)
                
                
            else: # use both original training AND estimator training data for training the estimator
                inputsEst = torch.tensor(np.concatenate((self.inputsTrainingEst,self.inputsTraining)), 
                                                         dtype=self.floatType).to(self.computeDevice)
                targetsEst = torch.tensor(np.concatenate((self.targetsTrainingEst,self.targetsTraining)), 
                                                         dtype=self.floatType).to(self.computeDevice)
                
                
                inputsTestEst = torch.tensor(np.concatenate((self.inputsTestEst,self.inputsTest)), 
                                                         dtype=self.floatType).to(self.computeDevice)
                targetsTestEst = torch.tensor(np.concatenate((self.targetsTestEst,self.targetsTest)), 
                                                     dtype=self.floatType).to(self.computeDevice)
                
            datasetEst = TensorDataset(inputsEst, targetsEst)
            dataloaderEst = DataLoader(datasetEst, batch_size=batchSize*2,
                                    shuffle=self.dataLoaderShuffle)
            datasetTestEst = TensorDataset(inputsTestEst, targetsTestEst)
            dataloaderTestEst = DataLoader(datasetTestEst, batch_size=batchSize*2, shuffle=self.dataLoaderShuffle)
            # for the paramizers of the error estimator a second optimizer is needed
            optimizerEst = torch.optim.Adam(self.nnModelEst.parameters(), lr=self.learningRate) # , weight_decay=1e-4)       
            self.GetEstType()                 
            if self.verboseMode > 0:
                print('\nTrain Error estimator')
                print('estimator works on: ' + 'input'*bool(not(self.estOnOutput)) + 
                                                   'output'*bool(self.estOnOutput))
            self.minLossEst, self.epochBestLossEst, self.nnModelEstBest  = np.inf, -1, None
            
            for epoch in range(self.maxEpochsEst):  
                # ************************************** training error estimator ************************************** %
                for inputs, targets in dataloaderEst:     
                    # Forward pass of nnModel
                    outputs = self.nnModel(inputs) / outputScaling
                    if self.estOnOutput: 
                        eEstimated = self.nnModelEst(outputs)
                    else: 
                        eEstimated = self.nnModelEst(inputs)
                    targets /= outputScaling
                    # calculate mean error error between estimation and ground truth
                    # note that the mean is calculated in dimension 1 which represents the timesteps
                    # eRef, myErrorMapping = mapError(torch.mean(outputs - targets, 1), mapType = 0 ) # learn mapping
                    
                    if False:     
                        eRef = torch.mean(torch.abs(outputs - targets), 1) # eRef is Mean Absolute Error
                    else: 
                        eRef = torch_rmse(outputs-targets, 1)
                        
                    if self.inputOutputSizeNN[1][1] > 1: 
                        # eRef = torch.norm(eRef, dim=1).reshape([-1,1])
                        eRef = torch_rmse(eRef, 1) # obtain euclidean distance for the case of multiple dimensions
                        
                    # maptype 0: no scaling; 1: [0,1]; 2: [-1, 1]
                    if bool(self.nnModelEst.mapErrorInfo): 
                        eRefMapped, _ = mapError(torch.abs(eRef), self.nnModelEst.mapErrorInfo['type'], self.nnModelEst.mapErrorInfo)
                        # if self.inputOutputSizeNN[1][1] > 1: 
                            
                        lossEst = self.lossFunctionEst(eEstimated, eRefMapped.view(-1,1))
                    else: 
                        lossEst = self.lossFunctionEst(eEstimated, eRef) # Root of MSE! 
                    # lossEst = self.lossFunctionEst(eEstimated, eDesired)
                    # Backward pass and optimization
                    optimizerEst.zero_grad() #(set_to_none=True)
                    lossEst.backward()
                    optimizerEst.step()

                currentLossEst = lossEst.item()
                if (epoch % self.lossLogInterval == 0) or (epoch == self.maxEpochs-1):
                    self.lossLogEst += [[epoch, currentLossEst]]
                    
                if self.verboseMode > 0: 
                    self.printLoss(currentLossEst, epoch, self.maxEpochsEst, epochPrintInterval, tStart2)
                # self.myErrorMapping = myErrorMapping
                
                # ************************************** validation ************************************** %
                if testEvaluationInterval > 0 and (epoch > 0 
                                                   and epoch%testEvaluationInterval == 0
                                                   or epoch == maxEpochsEst-1
                                                   or currentLoss < self.lossThreshold):
                    self.nnModelEst.eval() #switch network to evaluation mode
                    results = []
                    with torch.no_grad(): #avoid calculation of gradients:
                        for inputs, targets in dataloaderTestEst:
                            # Forward pass
                            outputs = self.nnModel(inputs) / outputScaling
                            if self.estOnOutput: 
                                eEstimated = self.nnModelEst(outputs)
                            else: 
                                eEstimated = self.nnModelEst(inputs)
                            targets /=  outputScaling
                            if False: 
                                eRef = torch.mean(torch.abs(outputs - targets), 1)
                            else: 
                                eRef = torch_rmse(outputs-targets, 1)
                            if self.inputOutputSizeNN[1][1] > 1: 
                                # eRef = torch.norm(eRef, dim=1).reshape([-1,1])
                                eRef = torch_rmse(eRef, 1)
                                
                            # if relevant, apply mapping of error
                            if bool(self.nnModelEst.mapErrorInfo): 
                                eRefMapped, _ = mapError(torch.abs(eRef), self.nnModelEst.mapErrorInfo['type'], self.nnModelEst.mapErrorInfo)
                                # if self.inputOutputSizeNN[1][1] > 1: 
                                #     eRefMapped = torch.norm(eRefMapped, dim=1).reshape([-1,1])
                                lossEst = self.lossFunctionEst(eEstimated, eRefMapped.view(-1,1))
                            else: 
                                lossEst = self.lossFunctionEst(eEstimated, eRef)
                                
                            # outputs = outputs.view(-1, *outputSize)  # Reshape the outputs to match the target shape
                            # mse = self.lossFunction(eEstimated, eRef)
                            results += [float(lossEst)]
                        self.lossValErrEst += [[epoch, np.mean(results)]] # mean results over 
                        if self.lossValErrEst[-1][1] < self.minLossEst: 
                            self.minLossEst = self.lossValErrEst[-1][1] 
                            self.nnModelBestEst = copy.deepcopy(self.nnModelEst)
                            self.epochBestLossEst = epoch
                            print('new best minimal loss: {} at epoch {}'.format(roundSignificant(self.minLossEst, 4), self.epochBestLossEst))
                            
                        
                        
                        self.nnModelEst.train() # set network into training mode
            print('{} estimator: {}'.format(self.lossFunctionEst, currentLossEst))
            if self.epochBestLossEst != epoch and self.epochBestLossEst != -1: 
                self.nnModelEst = copy.deepcopy(self.nnModelBestEst)
                print('using currently best model from validation at epoch ', self.epochBestLossEst)
            
            
            
            
        
        
                            
    #return loss recorded by TrainModel
    def LossLog(self):
        return np.array(self.lossLog)

    #return test results recorded by TrainModel
    def TestResults(self):
        return self.testResults

    #save trained model, which can be loaded including the structure (hidden layers, etc.)
    def SaveNNModel(self, fileName):
        os.makedirs(os.path.dirname(fileName), exist_ok=True)
        torch.save(self.nnModel, fileName) #save pytorch file
        if self.useErrorEstimator: 
            filenameEstimator = fileName.split('.')[0] + '_estimator.' + fileName.split('.')[1]
            torch.save(self.nnModelEst, filenameEstimator) #save pytorch file

            
    #load trained model, including the structure (hidden layers, etc.)
    def LoadNNModel(self, fileName):
        self.nnModel = torch.load(fileName)
        if self.useErrorEstimator: 
            filenameEstimator = fileName.split('.')[0] + '_estimator.' + fileName.split('.')[1]
            self.nnModelEst = torch.load(filenameEstimator)
            self.GetEstType() # read if Error estimator uses input or output of Surrogate network
            

    #plot loss and tests over epochs, using stored data from .npy file
    #dataFileTT is the training and test data file name
    def PlotTrainingResults(self, resultsFileNpy, dataFileName, plotLoss=True, plotTests=True, 
                            plotTestsMaxError=True, plotTestsMeanError=False,
                            sizeInches=[8,6], closeAll=True, plotEstimator = False,
                            ):
        
        def concatResults(epoch, resultsMax): 
            if np.array(resultsMax).shape != np.array(epoch).shape: 
                if np.array(resultsMax).size != np.array(epoch).size: 
                    print('warning PlotTrainingResults: \nShape of validation epochs and results does not match! Assuming geometrical data, ', 
                          'thus using euclidean norm in non-matching dimension.')
                    errMaxEuclidean = np.linalg.norm(resultsMax, axis=1)
                    dataTestMax = np.vstack((epoch, errMaxEuclidean)).T
                else: 
                    errMaxEuclidean = np.reshape(resultsMax, np.array(epoch).shape)
                    dataTestMax = np.vstack((epoch, errMaxEuclidean)).T
            else: 
                dataTestMax = np.vstack((epoch,resultsMax )).T
            return dataTestMax
        
        #load and visualize results:
        with open(resultsFileNpy, 'rb') as f:
            dataDict = np.load(f, allow_pickle=True).all()   #allow_pickle=True for lists or dictionaries; .all() for dictionaries

        self.LoadTrainingAndTestsData(dataFileName)

        if closeAll:
            PlotSensor(None,closeAll=True)

        values = dataDict['values']
        parameters = dataDict['parameters']
        parameterDict = dataDict['parameterDict']
        #plot losses:
        labelsList = []
        
        markerStyles = []
        markerSizes = [6]
        
        cntColor = -1
        lastCase = 0
        for i, v in enumerate(values):
            #colors are identical for cases
            if 'case' in parameters:
                case = parameterDict['case'][i]
                markerStyles=[listMarkerStyles[case]]
                if case < lastCase or cntColor < 0:
                    cntColor += 1
                lastCase = case
            else:
                cntColor += 1

            sLabel='var'+str(i)
            sep = ''
            for j,par in enumerate(list(parameters)):
                sLabel+=sep+VariableShortForm(par)+str(SmartRound2String(parameterDict[par][i],1))
                sep=','
            labelsList += [sLabel]
            data = np.vstack((v['lossEpoch'],v['loss'])).T
            
            if plotLoss:
                PlotSensor(None, [data], xLabel='epoch', yLabel='loss', labels=[sLabel],
                           newFigure=i==0, colorCodeOffset=cntColor, 
                           markerStyles=markerStyles, markerSizes=markerSizes, 
                           sizeInches=sizeInches, logScaleY=True)

        cntColor = -1
        lastCase = 0
        for i, v in enumerate(values):
            if 'case' in parameters:
                case = parameterDict['case'][i]
                markerStyles=[listMarkerStyles[case]]
                if case < lastCase or cntColor < 0:
                    cntColor += 1
                lastCase = case
            else:
                cntColor += 1

            sLabel = labelsList[i]
            if plotTests:
                dataTest = np.vstack((v['testResultsEpoch'],v['testResultsMean'] )).T
                dataTestMax = concatResults(v['testResultsEpoch'],v['testResultsMax'])
                
                if plotTestsMaxError:
                    PlotSensor(None, [dataTestMax], xLabel='epoch', yLabel='validation max MSE', labels=[sLabel+'(max)'],
                               newFigure=i==0, colorCodeOffset=cntColor, lineWidths=[1], 
                               markerStyles=markerStyles, markerSizes=markerSizes, 
                               sizeInches=sizeInches, logScaleY=True)

                if plotTestsMeanError:
                    PlotSensor(None, [dataTest], xLabel='epoch', yLabel='validation MSE ', labels=[sLabel+'(mean)'],
                               newFigure=(not plotTestsMaxError and i==0), colorCodeOffset=cntColor, lineWidths=[2], 
                               markerStyles=markerStyles, markerSizes=markerSizes, 
                               sizeInches=sizeInches, logScaleY=True)

        if 'lossEst' in values[0].keys(): 
            cntColor = -1
            lastCase = 0
            for i, v in enumerate(values):
                if 'case' in parameters:
                    case = parameterDict['case'][i]
                    markerStyles=[listMarkerStyles[case]]
                    if case < lastCase or cntColor < 0:
                        cntColor += 1
                    lastCase = case
                else:
                    cntColor += 1
    
                sLabel = labelsList[i]
                if plotTests:
                    dataEstimator = np.vstack((v['lossEpochEst'],v['lossEst'] )).T
                    # dataTestMax = np.vstack((v['testResultsEpoch'],v['testResultsMax'] )).T
                    
                    PlotSensor(None, [dataEstimator], xLabel='epoch', yLabel='MSE loss estimator', labels=[sLabel],
                               newFigure=i==0, colorCodeOffset=cntColor, lineWidths=[1], 
                               markerStyles=markerStyles, markerSizes=markerSizes, 
                               sizeInches=sizeInches, logScaleY=True)
                    
        if 'velErr' in values[0].keys(): 
            cntColor = -1
            lastCase = 0
            for i, v in enumerate(values):
                if 'case' in parameters:
                    case = parameterDict['case'][i]
                    markerStyles=[listMarkerStyles[case]]
                    if case < lastCase or cntColor < 0:
                        cntColor += 1
                    lastCase = case
                else:
                    cntColor += 1
    
                sLabel = labelsList[i]
                if plotTests:
                    dataEstimator = np.vstack((v['valEpochEst'],v['velErr'] )).T
                    # dataTestMax = np.vstack((v['testResultsEpoch'],v['testResultsMax'] )).T
                    
                    PlotSensor(None, [dataEstimator], xLabel='epoch', yLabel='validation error estimator', labels=[sLabel],
                               newFigure=i==0, colorCodeOffset=cntColor, lineWidths=[1], 
                               markerStyles=markerStyles, markerSizes=markerSizes, 
                               sizeInches=sizeInches, logScaleY=True)

            

    # evaluate all training and test data; plot some tests
    # nTrainingMSE avoids large number of training evaluations for large data sets
    def EvaluateModel(self, plotTests=[], plotTrainings=[], plotVars=['time','ODE2'], 
                      closeAllFigures=False, nTrainingMSE=64, measureTime=False,
                      saveFiguresPath='', figureEnding='.pdf', plotOptions={}, computeRMSE = False, 
                      flagErrorBarPlot = False):
        # self.nnModel.eval() #switch network to evaluation mode
        try: 
            mbs = self.simulationModel.mbs
            [inputSize, outputSize] = self.simulationModel.GetInputOutputSizeNN()
            outputScaling = self.simulationModel.GetOutputScaling().reshape(outputSize)

        except: 
            SC = exu.SystemContainer()
            mbs = SC.AddSystem() # for using PlotSensor function
            inputSize, outputSize = self.inputOutputSizeNN
            outputScaling = 1
            
        if len(plotTests)+len(plotTrainings) and closeAllFigures:
            mbs.PlotSensor(closeAll=True)

        outputScalingTensor = torch.tensor(outputScaling, dtype=self.floatType).to(self.computeDevice)
        

        inputs = torch.tensor(self.inputsTraining, dtype=self.floatType).to(self.computeDevice)
        targets = torch.tensor(self.targetsTraining, dtype=self.floatType).to(self.computeDevice)
        inputsTest = torch.tensor(self.inputsTest, dtype=self.floatType).to(self.computeDevice)
        targetsTest = torch.tensor(self.targetsTest, dtype=self.floatType).to(self.computeDevice)

        hiddenInit = torch.tensor(self.hiddenInitTraining, dtype=self.floatType).to(self.computeDevice)
        hiddenInitTest = torch.tensor(self.hiddenInitTest, dtype=self.floatType).to(self.computeDevice)


        dataset = TensorDataset(inputs, targets, hiddenInit)
        dataloader = DataLoader(dataset, batch_size=1)
        datasetTest = TensorDataset(inputsTest, targetsTest, hiddenInitTest)
        dataloaderTest = DataLoader(datasetTest, batch_size=1)
        
        # RMSE or RMSE?
        errSurrogate = {'train': [], 'test': [], 'trainEst': [], 'testEst': []}
        errEst       = {'train': [], 'test': [], 'trainEst': [], 'testEst': []}

        if self.useErrorEstimator: 
            inputsTestEst = torch.tensor(self.inputsTestEst, dtype=self.floatType).to(self.computeDevice)
            targetsTestEst = torch.tensor(self.targetsTestEst, dtype=self.floatType).to(self.computeDevice)
            datasetTestEst = TensorDataset(inputsTestEst, targetsTestEst)
            dataloaderTestEst = DataLoader(datasetTestEst, batch_size=1)
        
         
        trainingMSE = []
        testMSE = []
        testMSEEst = []
        trainRMSE = []
        testRMSE = []


        saveFigures = (saveFiguresPath != '')
        nMeasurements = 0
        timeElapsed = 0.
        self.nnModel.eval() #switch network to evaluation mode
        with torch.no_grad(): # avoid calculation of gradients
        
        ## -------------------------------- training set -------------------------------- ##
            newFigure=True
            i = -1
            
            for inputs, targets, initial_hidden_states in dataloader:
                i+=1
                if i < nTrainingMSE or (i in plotTrainings):
                    self.nnModel.SetInitialHiddenStates(initial_hidden_states)
                    if measureTime:
                        timeElapsed -= timer()
                        outputs = self.nnModel(inputs)
                        timeElapsed += timer()
                        nMeasurements += 1
                    else:
                        outputs = self.nnModel(inputs)
                        
                    if self.useErrorEstimator: 
                        if self.estOnOutput: 
                            errEstimated = self.nnModelEst(outputs)
                        else:  
                            errEstimated = self.nnModelEst(inputs)
                        if bool(self.nnModelEst.mapErrorInfo): # map back
                            errEstimated = mapErrorInv(errEstimated, self.nnModelEst.mapErrorInfo)
                        # errRef = torch.mean(outputs - targets, 1)
                        # errRef = torch.mean(torch.abs(outputs/outputScalingTensor - targets/outputScalingTensor), 1) 
                        errRef = torch_rmse(outputs/outputScalingTensor - targets / outputScalingTensor)
                        labelErr = f'| eEst={errEstimated[0][0]:.1e} ({errRef[0][0]:.1e})' # .format(roundSignificant(errEstimated[0][0], 2), roundSignificant(, 2))
                    else: 
                        labelErr = ''
                        
                    outputs = outputs.view(-1, *outputSize)  # Reshape the outputs to match the target shape
                    trainingMSE += [float(self.lossFunction(outputs, targets))]
                    trainRMSE += [np.sqrt(trainingMSE[-1])]
                    if self.useErrorEstimator: 
                        errSurrogate['train'] += [errRef[0][0]]
                        errEst['train'] += [errEstimated[[0][0]]]
                    
                if i in plotTrainings:
                    y = np.array(outputs.tolist()[0])
                    yRef = self.targetsTraining[i:i+1][0]
        
                    y = y/outputScaling
                    yRef = yRef/outputScaling
        
                    data = self.simulationModel.OutputData2PlotData(y)
                    dataRef = self.simulationModel.OutputData2PlotData(yRef)
    
                    comp = [self.simulationModel.PlotDataColumns()[plotVars[1]] - 1]#time=-1
                    compX = [self.simulationModel.PlotDataColumns()[plotVars[0]] - 1]#time=-1
                        
                    fileName = ''
                    #print('plotTrainings',plotTrainings,i,saveFigures)
                    if len(plotTrainings) > 0 and i==plotTrainings[-1] and saveFigures:
                        fileName = saveFiguresPath+plotVars[1]+'Training'+figureEnding
                    
                    if len(data) == 1: # quick and dirty fix for slidercrank: outputData has 1 dimension too much... 
                        data = data[0].T
                        dataRef = dataRef[0].T
                    # print('i: {} data shape: {}, dataRef shape: {}'.format(i, data.shape, dataRef.shape))
                    mbs.PlotSensor(data, components=comp, componentsX=compX, newFigure=newFigure,
                                    labels='NN train'+str(i) + labelErr, 
                                    xLabel = plotVars[0], yLabel=plotVars[1],
                                    colorCodeOffset=i%28, **plotOptions)
                    mbs.PlotSensor(dataRef, components=comp, componentsX=compX, newFigure=False,
                                    labels='Ref train'+str(i), 
                                    xLabel = plotVars[0], yLabel=plotVars[1],
                                    colorCodeOffset=i%28,lineStyles=[':'],
                                    fileName=fileName, **plotOptions)
                    newFigure = False

        
            newFigure=True
            
    ## -------------------------------- Validation set -------------------------------- ##
            i = -1
            for inputs, targets, initial_hidden_states in dataloaderTest:
                i+=1
                if not(i < nTrainingMSE or (i in plotTests)): 
                    continue
                self.nnModel.SetInitialHiddenStates(initial_hidden_states)
                outputs = self.nnModel(inputs)
                outputs = outputs.view(-1, *outputSize)  # Reshape the outputs to match the target shape
                testMSE += [float(self.lossFunction(outputs, targets))]
                if computeRMSE : 
                    # valRMSE = [float(self.lossFunction(outputs.cpu()/outputScaling, targets.cpu()/outputScaling))]
                    testRMSE += [np.sqrt(testMSE[-1])]
                if self.useErrorEstimator: 
                    if self.estOnOutput: 
                        errEstimated = self.nnModelEst(outputs)
                    else: 
                        errEstimated = self.nnModelEst(inputs)
                    if bool(self.nnModelEst.mapErrorInfo): # map back
                        errEstimated = mapErrorInv(errEstimated, self.nnModelEst.mapErrorInfo)
                    # errRef = torch.sqrt(torch.nn.MSELoss()(outputs, targets))
                    # errRef = torch.mean(torch.abs(outputs/outputScalingTensor - targets/outputScalingTensor), 1)  # mean absolute error; not used anymore
                    errRef = torch_rmse(outputs/outputScalingTensor - targets / outputScalingTensor)
                    # errRef = torch.nn.MSELoss()(outputs, targets)
                    # labelErr = '| eEst={} ({})'.format(roundSignificant(errEstimated[0][0], 2), roundSignificant(errRef, 2))
                    labelErr = f'| eEst={errEstimated[0][0]:.1e} ({errRef[0][0]:.1e})'
                else: 
                    labelErr = ''
                    
                if self.useErrorEstimator: 
                    errSurrogate['test'] += [errRef[0][0]]
                    errEst['test'] += [errEstimated[0][0]]
                
                if i in plotTests:
                    y = np.array(outputs.tolist()[0])
                    yRef = self.targetsTest[i:i+1][0]
                    
                    y = y/outputScaling
                    yRef = yRef/outputScaling
        
                    data = self.simulationModel.OutputData2PlotData(y)
                    dataRef = self.simulationModel.OutputData2PlotData(yRef)
    
                    comp = [self.simulationModel.PlotDataColumns()[plotVars[1]] - 1]#time=-1
                    compX = [self.simulationModel.PlotDataColumns()[plotVars[0]] - 1]
                        
                    fileName = ''
                    if len(plotTests) > 0 and i==plotTests[-1] and saveFigures:
                        fileName = saveFiguresPath+plotVars[1]+'Test'+figureEnding
                    
                    if len(data) == 1: # quick and dirty fix for slidercrank: outputData has 1 dimension too much... 
                        data = data[0].T
                        dataRef = dataRef[0].T
                        
                    mbs.PlotSensor(data, components=comp, componentsX=compX, newFigure=newFigure,
                                    labels='NN test'+str(i) + labelErr, 
                                    # labels='', 
                                    xLabel = plotVars[0], yLabel=plotVars[1],
                                    colorCodeOffset=i%28, **plotOptions)
                    mbs.PlotSensor(dataRef, components=comp, componentsX=compX, newFigure=False,
                                    labels='Ref test'+str(i), 
                                    # labels='', # 'Ref ' + str(i), 
                                    xLabel = plotVars[0], yLabel=plotVars[1],
                                    colorCodeOffset=i%28,lineStyles=[':'],
                                    fileName=fileName, **plotOptions)
                    newFigure = False
                    
            # errSurrogate['train'] = trainingMSE
            # errSurrogate['test'] = testMSE
            
        ## -------------------------------- Test set of Error Estimator-------------------------------- ##
            if self.useErrorEstimator: 
                newFigure=True
                i = -1

                for inputs, targets in dataloaderTestEst:
                    i+=1
                    if not(i < nTrainingMSE or (i in plotTests)): 
                        continue
                    outputs = self.nnModel(inputs)
                    # outputs = outputs.view(-1, *outputSize)  # Reshape the outputs to match the target shape
                    testMSEEst += [float(self.lossFunction(outputs, targets))]
                    if self.estOnOutput: 
                        errEstimated = self.nnModelEst(outputs)
                    else: 
                        errEstimated = self.nnModelEst(inputs)
                        
                    if bool(self.nnModelEst.mapErrorInfo): # map back
                        errEstimated = mapErrorInv(errEstimated, self.nnModelEst.mapErrorInfo)
                        
                    # errRef = self.lossFunction(outputs, targets)
                    # errRef = torch.mean(torch.abs(outputs/outputScalingTensor - targets/outputScalingTensor), 1) 
                    errRef = torch_rmse(outputs/outputScalingTensor - targets / outputScalingTensor)
                    errSurrogate['testEst'] += [errRef.cpu().detach().numpy()[0][0]]
                    errEst['testEst'] += [errEstimated.cpu().detach().numpy()[0][0]]
                    if False: 
                        errRef = torch.sqrt(errRef)
                        errEstimated = torch.sqrt(torch.abs(errEstimated))
                    # labelErr = '| eEst={} ({})'.format(roundSignificant(errEstimated[0][0], 3), roundSignificant(errRef, 3))
                    labelErr = f'| eEst={errEstimated[0][0]:.1e} ({errRef[0][0]:.1e})'

                        
                    if i in plotTests:
                        y = np.array(outputs.tolist()[0])
                        yRef = np.array(targets[0].cpu())
        
                        y = y/outputScaling
                        yRef = yRef/outputScaling
            
                        data = self.simulationModel.OutputData2PlotData(y)
                        dataRef = self.simulationModel.OutputData2PlotData(yRef)
            
                        comp = [self.simulationModel.PlotDataColumns()[plotVars[1]] - 1]#time=-1
                        compX = [self.simulationModel.PlotDataColumns()[plotVars[0]] - 1]
                            
                        fileName = ''
                        if len(plotTests) > 0 and i==plotTests[-1] and saveFigures:
                            fileName = saveFiguresPath+plotVars[1]+'Test'+figureEnding
                        
                        if len(data) == 1: # quick and dirty fix for slidercrank: outputData has 1 dimension too much... 
                            data = data[0].T
                            dataRef = dataRef[0].T
                            
                        mbs.PlotSensor(data, components=comp, componentsX=compX, newFigure=newFigure,
                                        labels='NN test'+str(i) + labelErr, xLabel = plotVars[0], yLabel=plotVars[1],
                                        colorCodeOffset=i%28, **plotOptions)
                        mbs.PlotSensor(dataRef, components=comp, componentsX=compX, newFigure=False,
                                        labels='Ref test'+str(i), xLabel = plotVars[0], yLabel=plotVars[1],
                                        colorCodeOffset=i%28,lineStyles=[':'],
                                        fileName=fileName, **plotOptions)
                        newFigure = False
                    plt.title('error estimator')
        
        def enforceArray(myData): 
            for i in range(len(myData)): 
                try: 
                    if type(myData[i][0]) is type(torch.tensor([])):
                        myData[i] = np.array(torch.tensor(myData[i]))
                except: 
                    pass
            return myData

        # if self.useErrorEstimator: 
        errSurrogate['train'] , errEst['train'], errSurrogate['test'], errEst['test'], errSurrogate['testEst'], errEst['testEst'] = enforceArray(\
        [errSurrogate['train'] , errEst['train'], errSurrogate['test'], errEst['test'], errSurrogate['testEst'], errEst['testEst']])    
            
        errSurrogate['train'] = np.array(errSurrogate['train'])
        errEst['train'] = np.array(errEst['train']).reshape(-1)
        
        errSurrogate['test'] = np.array(errSurrogate['test'])
        errEst['test'] = np.array(errEst['test'])
        
        
        errSurrogate['testEst'] = np.array(errSurrogate['testEst']) 
        errEst['testEst'] = np.array(errEst['testEst'])       
        


        if flagErrorBarPlot: 
            def plotError(errDictSurrogate, errDictEst, key): 
                n = len(errDictSurrogate[key])
                fact = 3
                plt.figure(key)
                if len(errDictEst[key]) > 0: # 
                    plt.bar(np.linspace(0, (n-1)*fact, n), errDictEst[key].reshape(-1), align='center', label='estimator')
                plt.bar(np.linspace(0, (n-1)*fact, n)+1, errDictSurrogate[key].reshape(-1), align='center', label='Surrogate')
                plt.xticks(np.linspace(0, (n-1)*fact, n//16+1, dtype=int)+0.5, np.linspace(0, (n-1), n//16 + 1, dtype=int))
                plt.grid()
                if len(errDictEst[key]) > 0: 
                    meanRelativeError = np.mean(np.abs(errDictEst[key] - errDictSurrogate[key])/ np.abs(errDictEst[key]))
                    plt.title('mean error estimator relative: '+ str(roundSignificant(meanRelativeError*100, 4)) + '%')
                plt.xlabel('dataset Error ' + str(key))
                plt.ylabel('mean absolute error')
                plt.legend()
            
            plotError(errSurrogate, errEst, 'train')
            plotError(errSurrogate, errEst, 'test')
            
            plotError(errSurrogate, errEst, 'testEst')
            
            
        if self.verboseMode > 0:
            print('max/mean test MSE=', roundSignificant(max(testMSE), 5), '/', roundSignificant(np.mean(testMSE), 5))
            print('max/mean training MSE=', roundSignificant(max(trainingMSE), 5), '/',roundSignificant(np.mean(trainingMSE), 5))
            if self.useErrorEstimator: 
                print('error estimator')
                print()
        if measureTime and nMeasurements>0:
            print('forward evaluation total CPU time:', SmartRound2String(timeElapsed))
            print('Avg. CPU time for 1 evaluation:', SmartRound2String(timeElapsed/nMeasurements))
        if computeRMSE: 
            return {'testMSE':testMSE, 'maxTrainingMSE':max(trainingMSE), 'testRMSE': testRMSE, 'testMSEEst': testMSEEst}
        else: 
            return {'testMSE':testMSE, 'maxTrainingMSE':max(trainingMSE)}
    
    # Utility function of the neural network training center for plotting error estimator performance
    def EvaluateModelEstimator(self, nMax = 128, flagPlotBestWorst = False, flagPlotCorreleation=False, 
                               factorUnit = 1): 
        def plotError(errDictSurrogate, errDictEst, key): 
            n = len(errDictSurrogate[key])
            fact = 3
            plt.figure(key)
            if len(errDictEst[key]) > 0: # 
                plt.bar(np.linspace(0, (n-1)*fact, n), errDictEst[key].reshape(-1), align='center', label='estimator')
            plt.bar(np.linspace(0, (n-1)*fact, n)+1, errDictSurrogate[key].reshape(-1), align='center', label='Surrogate')
            plt.xticks(np.linspace(0, (n-1)*fact, n//16+1, dtype=int)+0.5, np.linspace(0, (n-1), n//16 + 1, dtype=int))
            plt.grid()
            if len(errDictEst[key]) > 0: 
                meanRelativeError = np.mean(np.abs(errDictEst[key] - errDictSurrogate[key])/ np.abs(errDictEst[key]))
                plt.title('mean error estimator relative: '+ str(roundSignificant(meanRelativeError*100, 4)) + '%')
            plt.xlabel('dataset Error ' + str(key))
            plt.ylabel('mean absolute error')
            plt.legend()
            return
        def getMeanStdPercentile(data, percentile=95): 
            if torch.is_tensor(data): 
                data = data.detach().cpu().numpy()
            absData = np.abs(data)
            
            mean = np.mean(absData)
            std = np.std(absData)
            perc = np.percentile(absData, percentile)
            
            
            minAbs, iMin = np.min(absData), np.argmin(absData)
            maxAbs, iMax = np.max(absData), np.argmax(absData)
            return {'mean': mean, 'std': std, 'percentile': perc, 'minMax': [minAbs, maxAbs], 'i_minMax': [iMin, iMax]}
            
        def printDict(myDict): 
            myStr = ''
            for key, value in myDict.items(): 
                myStr += "{}: {}\n".format(key, roundSignificant(value, 3))
            print(myStr)
            
        inputs = torch.tensor(self.inputsTraining, dtype=self.floatType).to(self.computeDevice)
        targets = torch.tensor(self.targetsTraining, dtype=self.floatType).to(self.computeDevice)
        inputsTest = torch.tensor(self.inputsTest, dtype=self.floatType).to(self.computeDevice)
        targetsTest = torch.tensor(self.targetsTest, dtype=self.floatType).to(self.computeDevice)
        
        
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=len(inputs))
        datasetTest = TensorDataset(inputsTest, targetsTest)
        dataloaderTest = DataLoader(datasetTest, batch_size=len(inputs))
        
        # 
        inputOutputSizeNN = self.simulationModel.GetInputOutputSizeNN()
        outputScaling = self.simulationModel.GetOutputScaling().reshape(inputOutputSizeNN[1])
        outputScaling = torch.tensor(outputScaling).to(self.computeDevice)
        
        # RMSE or RMSE?
        errSurrogate = {'train': [], 'test': [], 'trainEst': [], 'testEst': []}
        errEst       = {'train': [], 'test': [], 'trainEst': [], 'testEst': []}
        trainError   = {'mean': 0, 'std': 0, 'percentile95': 0, }
        
        if not(self.useErrorEstimator): 
            raise ValueError('ERROR: NeuralNetworkTrainingCenter can not test estimator if it is not initialized.')
            
            
        inputsTrainEst = torch.tensor(self.inputsTrainingEst, dtype=self.floatType).to(self.computeDevice)
        targetsTrainEst = torch.tensor(self.targetsTrainingEst, dtype=self.floatType).to(self.computeDevice)
        datasetTrainEst = TensorDataset(inputsTrainEst, targetsTrainEst)
        dataloaderTrainEst = DataLoader(datasetTrainEst, batch_size=len(inputsTrainEst))
        
        inputsTestEst = torch.tensor(self.inputsTestEst, dtype=self.floatType).to(self.computeDevice)
        targetsTestEst = torch.tensor(self.targetsTestEst, dtype=self.floatType).to(self.computeDevice)
        datasetTestEst = TensorDataset(inputsTestEst, targetsTestEst)
        dataloaderTestEst = DataLoader(datasetTestEst, batch_size=len(inputsTestEst))
        
         
        trainingMSE = []
        testMSE = []
        testMSEEst = []
        trainRMSE = []
        testRMSE = []
        
        if flagPlotCorreleation: 
            figCorr, axsCorr = plt.subplots(1,2)
            axsCorr = axsCorr.reshape(2)
            iCorr = 0
            minErr, maxErr, perc = np.inf, 0, []
            R = [] # correlation coefficient 
            
        # saveFigures = (saveFiguresPath != '')
        nMeasurements = 0
        timeElapsed = 0.
        self.nnModel.eval() #switch network to evaluation mode
        self.nnModelEst.eval()
        with torch.no_grad(): # avoid calculation of gradients
        ## -------------------------------- training set -------------------------------- ##
            newFigure=True
            i = -1
            myError, errorList = {}, {}
            for dataSetType, myDataloader in zip(['train', 'test', 'trainEst', 'testEst'], [dataloader, dataloaderTest,dataloaderTrainEst, dataloaderTestEst]): 
                for inputs, targets in myDataloader:
                    outputs = self.nnModel(inputs) # in Manuscript: y hat
                    
                    # outputs = outputs / outputScaling
                    # targets_scaled = targets / outputScaling
                    
                    # outputs = outputs / outputScaling
                    if self.estOnOutput: 
                        errEstimated = self.nnModelEst(outputs)
                    else: 
                        errEstimated = self.nnModelEst(inputs)
                        
                    if bool(self.nnModelEst.mapErrorInfo): #  map errorback if it exists!
                        errEstimated = mapErrorInv(errEstimated, self.nnModelEst.mapErrorInfo)
                    # before calculating the reference error you MUST scale it back into the "original physical" space of the simulation model
                    # errRef = torch.mean(torch.abs(outputs/outputScaling - targets/outputScaling), 1) 
                    errRef = torch_rmse(outputs/outputScaling - targets/outputScaling, 1)
                    if self.simulationModel.GetInputOutputSizeNN()[1][1] > 1: 
                        # errRef = torch.norm(errRef, dim=1).reshape([-1,1])
                        errRef = torch_rmse(errRef, 1).view(-1,1)
                        
                    errRef_np  = errRef.clone().cpu().detach().numpy()
                    myRelativeError = ((errEstimated - errRef)/errRef)*100
                    myError[dataSetType] = getMeanStdPercentile(myRelativeError, percentile=95)
                    print('\ncalculating Error on dataset', dataSetType)
                    printDict(myError[dataSetType])
                    
                    errorList[dataSetType] = {}
                    errorList[dataSetType]['ref'] = errRef_np  
                    errorList[dataSetType]['est'] = errEstimated.clone().cpu().detach().numpy()
                    
                    # barplot of errors: 
                    n = len(errRef)
                    if n > nMax: n = nMax
                    
                    fact = 3
                    plt.figure('estimator-' + dataSetType)
                    xSteps = int(nMax / 8)
                    if factorUnit == 1000: 
                        strUnit = 'mm'
                    elif factorUnit == 1: 
                        strUnit = 'm'
                    
                    plt.bar(np.linspace(0, (n-1)*fact, n)+1, errorList[dataSetType]['ref'][:nMax].reshape(-1)*factorUnit, align='center', label='Surrogate')
                    plt.bar(np.linspace(0, (n-1)*fact, n)+2, errorList[dataSetType]['est'][:nMax].reshape(-1)*factorUnit, align='center', label='Estimator')
                    plt.xticks(np.linspace(0, (n-1)*fact, n//xSteps+1, dtype=int)+0.5, np.linspace(0, (n-1), n//xSteps + 1, dtype=int))

                    
                    plt.xlabel('dataset Error ' + str(dataSetType))
                    plt.ylabel('RMSE in mm')
                    plt.legend()
                    plt.grid()
                    strTitle = 'RMSE relative:\n'
                    strTitle += r'$\mu={}$% $\sigma={}$%, 95 percentile = ${}$%'.format(roundSignificant(myError[dataSetType]['mean'], 2), 
                                                                                  roundSignificant(myError[dataSetType]['std'], 3), 
                                                                                  roundSignificant(myError[dataSetType]['percentile'], 3))
                    
                    plt.title(strTitle, fontsize=14)
                    plt.gcf().set_size_inches(10, 6)
                    plt.tight_layout()
                    
                    if flagPlotBestWorst: 
                        # plt.figure('minMax ' + dataSetType)
                        fig, axs = plt.subplots(outputScaling.shape[1], 2)
                        col = ['g', 'r']
                        comment = ['best ', 'worst ']
                        for k, j in enumerate(myError[dataSetType]['i_minMax']): 
                            errors = [f'{float(errEstimated[j]):.2e}', f'{float(errRef[j]):.2e}', f'{round(float(myRelativeError[j]), 1)}']
                            
                            print(comment[k] + 'estimator: estimated: {} | ref: {} | diff in %: {}'.format(errors[0], errors[1], errors[2]))
                            data = (outputs[j] / outputScaling).detach().cpu().numpy()
                            dataRef = (targets[j] / outputScaling).detach().cpu().numpy()
                            for l in range(outputScaling.shape[1]): 
                                axs[l,0].plot(data[:,l], color=col[k], label=r'$\hat{e}=' + '{}$, $e={}$'.format(errors[0], errors[1]))
                                axs[l,0].plot(dataRef[:,l], '--', color=col[k], label='')
                        for l in range(outputScaling.shape[1]): 
                            axs[l,0].grid(True)
                            axs[l,0].legend()
                            
                        # plt.suptitle('best error: ' + str() + ', worst error: ' + str())
                        axs[0,0].set_title('Estimator min/max error')
                        
                        
                        for k, j in enumerate([np.argmin(errRef_np), np.argmax(errRef_np)]): 
                            errors = [f'{float(errEstimated[j]):.2e}', f'{float(errRef[j]):.2e}', f'{round(float(myRelativeError[j]), 1)}']
                            print(comment[k] + ' Surrogate: estimated: {} | ref: {} | diff in %: {}'.format(errors[0], errors[1], errors[2]))
                            data = (outputs[j] / outputScaling).detach().cpu().numpy()
                            dataRef = (targets[j] / outputScaling).detach().cpu().numpy()
                            for l in range(outputScaling.shape[1]): 
                                axs[l,1].plot(data[:,l], color=col[k], label=r'$\hat{e}=' + '{}$, $e={}$'.format(errors[0], errors[1]))
                                axs[l,1].plot(dataRef[:,l], '--', color=col[k])
                        for l in range(outputScaling.shape[1]): 
                            axs[l,1].grid(True)
                            axs[l,1].legend()
                            
                        axs[0,1].set_title('Surrogate model min/max error')

                        myYStr = ['x', 'y', 'z']
                        for l in range(outputScaling.shape[1]): 
                            yMin, yMax = +1e8, -1e8
                            for ax in axs[l,:]: 
                                yLim_i = ax.get_ylim() 
                                if yLim_i[0] < yMin: 
                                    yMin = yLim_i[0]
                                if yLim_i[1] > yMax: 
                                    yMax = yLim_i[1] 
                            for ax in axs[l,:]: 
                                ax.set_ylim([yMin, yMax])
                            axs[l,0].set_ylabel(f'deflection {myYStr[l]} in m')
                        for k in range(2): 
                            axs[-1,k].set_xlabel('steps')
                            
                    if flagPlotCorreleation and 'Est' in dataSetType: 
                        axsCorr[iCorr].plot(errorList[dataSetType]['ref']*factorUnit, errorList[dataSetType]['est']*factorUnit, 'x', markersize=2)
                        if dataSetType == 'trainEst': 
                            nTrain = len(errorList[dataSetType]['ref'])
                            
                        axsCorr[iCorr].set_ylabel(r'$\hat{e}$ in ' + strUnit)
                        axsCorr[iCorr].set_xlabel('$e$ in ' + strUnit)
                        
                        perc += [np.percentile(errorList[dataSetType]['ref']*factorUnit, 95)]
                        print(' true value percentile: ', round(perc[-1], 4))
                        
                        
                        axsCorr[iCorr].plot([0,1], [0,1], '--', color=[0.2]*3, alpha=0.4)
                        if dataSetType == 'trainEst': 
                            axsCorr[iCorr].set_title('training dataset estimator')
                        elif dataSetType == 'testEst':     
                            axsCorr[iCorr].set_title('test dataset estimator')
                        iCorr += 1                
                        minErr = min(minErr, min(np.min(errorList[dataSetType]['est']), np.min(errorList[dataSetType]['ref'])))
                        maxErr = max(maxErr, max(np.max(errorList[dataSetType]['est']), np.max(errorList[dataSetType]['ref'])))
                        R += [np.corrcoef(errorList[dataSetType]['ref'].reshape(-1), errorList[dataSetType]['est'].reshape(-1))] 
                        
            if flagPlotCorreleation: 
                for iCorr in range(axsCorr.size): 
                    axsCorr[iCorr].set_ylim([minErr*0.95*factorUnit, maxErr*1.05*factorUnit])
                    axsCorr[iCorr].set_xlim([minErr*0.95*factorUnit, maxErr*1.05*factorUnit])
                    axsCorr[iCorr].plot([perc[iCorr]]*2, [0,minErr*factorUnit], 'r-', label='95th percentile e', alpha=0.5)
                    axsCorr[iCorr].text(maxErr*factorUnit*0.29, minErr*factorUnit*1.1, 'R = {}'.format(round(R[iCorr][0,1], 3)))
                    axsCorr[iCorr].set_xscale("log")
                    axsCorr[iCorr].set_yscale("log")
                    axsCorr[iCorr].grid(True)
                    axsCorr[iCorr].set_aspect('equal', 'box')
                    axsCorr[iCorr].set_xticks([0.05, 0.1, 0.2, 0.4, 0.8], [0.05, 0.1, "0.2 ", 0.4, 0.8])
                    axsCorr[iCorr].set_yticks([0.05, 0.1, 0.2, 0.4, 0.8], [0.05, 0.1, 0.2,  0.4, 0.8])
                    # axsCorr[iCorr].legend()
                figCorr.set_size_inches(10, 5)
                figCorr.tight_layout()
                
                figCorr.text(0.315, 0.122, '$e_{95}$', color='r')
                figCorr.text(0.312 + 0.5, 0.122, '$e_{95}$', color='r')
                # plt.
                # plt.
                # axsCorr[0].text(0,0, 'ab')
                
                    
#%% test functions. Just used to see if functions can be called and dimension / 
# tensor shapes are correct. 
# The tests can be called using pytest with "!python -m pytest" from the console. 
def test_FFN(): 
    t1 = time.time()
    simModel = NonlinearOscillator(useVelocities=True, useInitialValues=True, 
                                nStepsTotal=40, endTime=0.25)
    
    nnModel = MyNeuralNetwork(inputOutputSize = simModel.GetInputOutputSizeNN(), # input and output size, 
                               neuralNetworkTypeName = 'FFN',
                               hiddenLayerSize = 40,
                               hiddenLayerStructure = ['L', 'R'],
                               computeDevice='cpu', 
                               )
    
    
    nntc = NeuralNetworkTrainingCenter(simulationModel=simModel, nnModel = nnModel, computeDevice='cpu')
    nntc.CreateTrainingAndTestData(nTraining=6, nTest=3
                                   )
    
    nntc.TrainModel(maxEpochs=1)
    t2 = time.time() 
    assert (t2-t1 < 2)
    print('FFN base test successful')
    return
    

def test_CNN(): 
    t1 = time.time()
    simModel = NonlinearOscillator(useVelocities=False, useInitialValues=False, 
                                nStepsTotal=40, endTime=0.25, variationMKD=False)
    
    nnModel = MyNeuralNetwork(inputOutputSize = simModel.GetInputOutputSizeNN(), # input and output size, 
                               neuralNetworkTypeName = 'CNN',
                               hiddenLayerSize = 40,
                               hiddenLayerStructure = ['L', 'R', 'L'],
                               computeDevice='cpu', 
                               )

    nntc = NeuralNetworkTrainingCenter(simulationModel=simModel, nnModel = nnModel, computeDevice='cpu')
    
    nntc.CreateTrainingAndTestData(nTraining=6, nTest=3
                                   )
    
    nntc.TrainModel(maxEpochs=1)
    t2 = time.time() 
    assert (t2-t1 < 2)
    print('CNN base test successful')    
    return


def test_errorEstimator(): 
    t1 = time.time()
    simModel = NonlinearOscillator(useVelocities=False, useInitialValues=False, 
                                nStepsTotal=40, endTime=0.25)
    
    nnModel = MyNeuralNetwork(inputOutputSize = simModel.GetInputOutputSizeNN(), # input and output size, 
                               neuralNetworkTypeName = 'CNN',
                               hiddenLayerSize = 40,
                               hiddenLayerStructure = ['L', 'R', 'L'],
                               computeDevice='cpu', 
                               )
    
    nnModelEst = MyNeuralNetwork(inputOutputSize = simModel.GetInputOutputSizeNN(), # input and output size, 
                               neuralNetworkTypeName = 'FFN',
                               hiddenLayerSize = 20,
                               hiddenLayerStructure = ['L', 'R', 'L'],
                               computeDevice='cpu', 
                               )
    
    nntc = NeuralNetworkTrainingCenter(simulationModel=simModel, nnModel = nnModel, computeDevice='cpu', nnModelEst=nnModelEst)
    nntc.CreateTrainingAndTestData(nTraining=6, nTest=3)
    
    nntc.TrainModel(maxEpochs=5)
    t2 = time.time() 
    assert (t2-t1 < 5)
    print('error estimator base test successful')
    

#%% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__': #include this to enable parallel processing
    
    torch.manual_seed(42)
    np.random.seed(42)
    test_CNN()
    test_FFN()
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #put this part out of if __name__ ... if you like to use multiprocessing
    endTime=0.25
    useErrorEstimator = True
    
    # calling 
    simModel = NonlinearOscillator(useVelocities=False, useInitialValues=True, 
                                nStepsTotal=64*2, endTime=endTime*5, variationMKD=False, flagDuffing=True, ) 
    
        
    simModelSCFlex = SliderCrank(nStepsTotal=200, nCutInputSteps = 50, nOutputSteps = 50, tStartup = 0.5, endTime= 2, useInitialVelocities=False, useInitialAngles=True, 
                                 useTorqueInput=True, flagFlexible=True, 
                                 initAngleRange = [-np.pi,np.pi], initVelRange  = [-12,12])
    simModelSCFlex.CreateModel()
    
    nnModel = MyNeuralNetwork(inputOutputSize = simModelSCFlex.GetInputOutputSizeNN(), # input and output size, 
                               neuralNetworkTypeName = 'CNN',
                               hiddenLayerSize = 60,
                               hiddenLayerStructure = ['L', 'R', 'L'],
                               computeDevice='cpu', 
                               )

    
    nnModelEst = None
    if useErrorEstimator: 
        nnModelEst = CreatePresetNeuralNetwork('FFN', 'LRL', np.prod(simModel.GetInputOutputSizeNN()[0]), 1, 60)
        
    #MyNeuralNetwork()
    nntc = NeuralNetworkTrainingCenter(simulationModel=simModelSCFlex, nnModel = nnModel, computeDevice='cpu', 
                                       nnModelEst=nnModelEst, inputOutputSizeNN=simModelSCFlex.GetInputOutputSizeNN())
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    nntc.CreateTrainingAndTestData(nTraining=12, nTest=4, 
                                    # parameterFunction=PVCreateData, #for multiprocessing
                                   )
    moduleNntc = nntc

    nntc.TrainModel(maxEpochs=500, batchSize=64)
    
    
    nntc.EvaluateModel(plotTests=[0,1,2,3,4], plotTrainings=[0,1,2,3,4], plotVars=['time','ODE2'])
