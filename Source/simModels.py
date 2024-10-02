#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  Library for creating multibody simulation models for the SLIDE method. 
#
# Author:   Peter Manzl, Johannes Gerstmayr
# Date:     2024-09-28
#
# Copyright: See Licence.txt
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import exudyn as exu
from exudyn.utilities import *
from exudyn.signalProcessing import GetInterpolatedSignalValue, FilterSignal
from exudyn.physics import StribeckFunction
from exudyn.utilities import SensorNode, SensorObject, Rigid2D, RigidBody2D, Node1D, Mass1D, GraphicsDataSphere, color4black
from exudyn.utilities import ObjectGround, VObjectGround, GraphicsDataOrthoCubePoint, color4grey, VCoordinateSpringDamper
from exudyn.utilities import NodePointGround, Cable2D, MarkerBodyPosition, MarkerBodyRigid, VMass1D, color4blue, MarkerNodeCoordinate
from exudyn.utilities import CoordinateSpringDamper, VObjectRigidBody2D, CoordinateConstraint, MarkerNodePosition, Point2D
from exudyn.utilities import RevoluteJoint2D, GraphicsDataRectangle, SensorUserFunction, Torque, Force, LoadCoordinate, copy
from exudyn.utilities import MassPoint2D, VCable2D, GenerateStraightLineANCFCable2D, color4dodgerblue, SensorBody, RotationMatrixZ
from exudyn.FEM import *

import random
from exudyn.robotics import Robot, RobotBase, RobotLink, RobotTool, StdDH2HT, VRobotBase, VRobotTool, VRobotLink
from exudyn.robotics.motion import Trajectory, ProfileConstantAcceleration, ProfilePTP

import sys
import numpy as np
from math import sin, cos, pi, tan, exp, sqrt, atan2

from enum import Enum #for data types
import time

try: 
    import exudyn.graphics as graphics 
except: 
    print('exudyn graphics could not be loaded. Make sure a version >= 1.8.52 is installed.' )
    print('some simulation models may not work correctly. ')
    
try: 
    import ngsolve as ngs
    import netgen
    from netgen.meshing import *
    from netgen.geom2d import unit_square
    from netgen.csg import *
except: 
    print('warning: ngsolve/netgen could not be loaded, thus the flexible robot model does not work.')
    
class ModelComputationType(Enum):
    dynamicExplicit = 1         #time integration
    dynamicImplicit = 2         #time integration
    static = 3         #time integration
    eigenAnalysis = 4         #time integration

    #allows to check a = ModelComputationType.dynamicImplicit for a.IsDynamic()    
    def IsDynamic(self):
        return (self == ModelComputationType.dynamicExplicit or 
                self == ModelComputationType.dynamicImplicit)


def AccumulateAngle(phi): 
    phiCorrected = copy.copy(phi)
    for i in range(1, len(phiCorrected)): 
        if abs(phiCorrected[i] - phiCorrected[i-1]) > np.pi: 
            phiCorrected[i:] += 2*np.pi * np.sign(phiCorrected[i-1] - phiCorrected[i])
    return phiCorrected

# 
def CreateVelocityProfile(tStartup, tEnd, nStepsTotal, vMax, aMax, nPeriods = [20, 60], flagNoise=True, trajType = 0): 
    
    v = np.zeros(nStepsTotal)
    h = tEnd / nStepsTotal
    nStartup = int(np.ceil(tStartup/h))
    v_start = np.random.uniform(-vMax, vMax)
    v[0:nStartup] = np.linspace(0, v_start, nStartup)    
    i_0 = nStartup
    i_E = i_0

    # if activated 50% chance to add noise
    if trajType  == 0: 
        if np.random.random() > 0.5 and flagNoise: 
            addNoise = True
        else: 
            addNoise = False
            
        while i_E < nStepsTotal: 
            i_E += int(np.random.uniform(nPeriods[0], nPeriods[1]))
            # print('idiff: ', i_diff)
            if i_E > nStepsTotal: 
                i_E = nStepsTotal
            di = i_E - i_0
            dt = di * h
            a_ = np.random.uniform(-aMax, aMax)
            if np.random.random() < 0.1:  # 10% are constant
                a_ = 0
                
            dv = a_*dt
            if abs(v[i_0-1] + dv) > vMax: 
                dv = vMax * np.sign(v[i_0-1] + dv)  - v[i_0-1]
                if a_ == 0: 
                    continue
                
                dt = dv / a_                
                i_E = i_0 + int(dt / h)
                if dt < h*10: 
                    continue
                if i_E > nStepsTotal: 
                    i_E = nStepsTotal
            if dt > nPeriods[1]*h: 
                dt = nPeriods[1]*h
            if (i_E - i_0 + 1) < 0: 
                continue
            v[i_0-1:i_E] = v[i_0-1] + np.linspace(0, a_*dt, i_E - i_0 +1)
            maxNoise = 0
            if addNoise: 
                maxNoise = 0.04 * vMax
                nVal = len(v[i_0-1:i_E])
                v[i_0-1:i_E] += ((np.random.random(nVal) - 0.5)* maxNoise) * (np.random.random(nVal) > 0.5)
            if abs(v[i_E-1]) > (vMax + bool(flagNoise) * maxNoise + 1e-12): 
                if a_ == 0:
                    continue
                i_max = i_0 + (vMax - v[qi_0]) / a_
                print('Velocity too high at {}: {}'.format(i_E, v[i_E-1]))
            # print('traj length: {} steps'.format(i_E - i_0))
            i_0 = i_E
        v = np.clip(v, -vMax, vMax) # avoid values above vMax
        
    elif trajType  == 1: 
        v[i_0:i_0+nStartup] = v[i_0-1]
        i_0 = i_0+nStartup 
        n2 = int(np.ceil(np.random.uniform(nPeriods[0], nPeriods[1])))
        i_E = i_0 + n2
        v2 = np.random.uniform(-vMax, vMax)
        dt2 = n2 * tEnd/nStepsTotal
        if abs(v[i_0-1] - v2) > aMax * dt2:
            v2 = v[i_0-1] + np.sign(v2 - v[i_0-1]) * dt2 * aMax
        v[i_0:i_E] = np.linspace(v[i_0-1], v2, n2)
        v[i_E:] = v[i_E-1]
        if flagNoise: 
            maxNoise = 0.04 * vMax
            v += ((np.random.random(nStepsTotal) - 0.5)* maxNoise) * (np.random.random(nStepsTotal) > 0.5)
        
    else: 
        raise ValueError('CreateVelocityProfile does not support trajType {}'.format(trajType ))
    return v


# %timeit numericIntegrate(v, dt): 
# 33.9 µs ± 233 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
# According to timeit faster than zero allocation (38 µs ± 141 ns per loop) and
# using np.tri version (72.2 µs ± 1.79 µs per loop); probably because of big 
# memory allocation for larger v vectors. Note that for shorter vectors of v
# the version with the np.tri becomes faster: for len(v) == 100
# 17.1 µs ± 156 ns --> 11.6 µs ± 88 ns.
def numericIntegrate(v, dt): 
    p = [0]
    for i in range(1, len(v)): 
        p += [p[-1] + v[i] * dt]
    return p
    # return np.tri(len(v), len(v)) @ v * dt


#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++
# base class for creating an Exudyn model
class SimulationModel():

    #initialize class 
    def __init__(self):
        self.SC = None
        self.mbs = None
        self.modelName = 'None'
        self.modelNameShort = 'None'
        self.inputScaling = np.array([])
        self.outputScaling = np.array([])
        self.inputScalingFactor = 1. #additional scaling factor (hyper parameter)
        self.outputScalingFactor = 1.#additional scaling factor (hyper parameter)
        self.nOutputDataVectors = 1  #number of output vectors (e.g. x,y,z)
        self.nnType = 'unused' #  this is currently not used. The simulation model should be seperated from the neural network model. 
        
        self.nStepsTotal = None
        self.computationType = None
        #number of ODE2 states
        # ODE2 states are coordinates which are describes by differential equations of second order. 
        self.nODE2 = None               
        self.useVelocities = None
        self.useInitialValues = None
        
        self.simulationSettings = None
    
    # warning: depreciated!     
    def NNtype(self):
        return self.nnType
    def IsRNN(self):
        return self.nnType == 'RNN'
    def IsFFN(self):
        return self.nnType == 'FFN'
    def IsCNN(self): 
        return self.nnType == 'CNN'
    
    #create a model and interfaces
    def CreateModel(self):
        pass

    #get model name
    def GetModelName(self):
        return self.modelName

    #get short model name
    def GetModelNameShort(self):
        return self.modelNameShort

    #get number of simulation steps
    def GetNSimulationSteps(self):
        return self.nStepsTotal

    #return a numpy array with additional scaling for inputs when applied to mbs (NN gets scaled data!)
    #also used to determine input dimensions
    def GetInputScaling(self):
        return self.inputScalingFactor*self.inputScaling
    
    #return a numpy array with scaling factors for output data
    #also used to determine output dimensions
    def GetOutputScaling(self):
        return self.outputScalingFactor*self.outputScaling

    #return input/output dimensions [size of input, shape of output]
    def GetInputOutputSizeNN(self):
        return [self.inputScaling.shape, self.outputScaling.shape]
    
    #get time vector according to output data
    def GetOutputXAxisVector(self):
        return np.array([])

    #create a randomized input vector
    #relCnt can be used to create different kinds of input vectors (sinoid, noise, ...)
    #isTest is True in case of test data creation
    def CreateInputVector(self, relCnt = 0, isTest=False, dataErrorEstimator = False):
        return np.array([])

    #create initialization of (couple of first) hidden states
    def CreateHiddenInit(self, isTest):
        return np.array([])
    
    #split input data into initial values, forces or other inputs
    #return dict with 'data' and possibly 'initialODE2' and 'initialODE2_t'
    def SplitInputData(self, inputData, hiddenData=None):
        return {'data':None}
    
    #split output data to get ODE2 values (and possibly other data, such as ODE2)
    #return dict {'ODE2':[], 'ODE2_t':[]}
    def SplitOutputData(self, outputData):
        return {'ODE2':[]}

    #convert all output vectors into plottable data (e.g. [time, x, y])
    #the size of data allows to decide how many columns exist
    def OutputData2PlotData(self, outputData, forSolutionViewer=False):
        return np.array([])

    #return dict of names to columns for plotdata        
    def PlotDataColumns(self):
        return {}

    #get compute model with given input data and return output data
    def ComputeModel(self, inputData, hiddenData=None, verboseMode = 0, solutionViewer = False):
        return np.array([])
    
    #visualize results based on given outputData
    #outputDataColumns is a list of mappings of outputData into appropriate column(s), not counting time as a column
    #  ==> column 0 is first data column
    def SolutionViewer(self, outputData, outputDataColumns = [0]):
        nColumns = self.nODE2
        data = self.OutputData2PlotData(outputData, forSolutionViewer=True)
        
        # columnsExported = dict({'nODE2':self.nODE2, 
        #                         'nVel2':0, 'nAcc2':0, 'nODE1':0, 'nVel1':0, 'nAlgebraic':0, 'nData':0})
        columnsExported = [nColumns, 0, 0, 0, 0, 0, 0] #nODE2 without time
        if data.shape[1]-1 != nColumns:
            raise ValueError('SimulationModel.SolutionViewer: problem with shape of data: '+
                             str(nColumns)+','+str(data.shape))

        nRows = data.shape[0]
        
        
        sol = dict({'data': data, 'columnsExported': columnsExported,'nColumns': nColumns,'nRows': nRows})

        self.mbs.SolutionViewer(sol,runOnStart=True)
    

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class NonlinearOscillator(SimulationModel):
    #initialize class 
    def __init__(self, nStepsTotal=100, useVelocities=True, useInitialValues=True, useFriction=False,
                 nMasses=1, endTime=1, variationMKD=False,  flagDuffing=False, nonlinearityFactor = 0.05, 
                 nStepForces=1, useHarmonicExcitation=False, useRandomExcitation=False, initOnlyLast = True, 
                 flagNoForce = False):
        SimulationModel.__init__(self)

        #required in base:
        self.nStepsTotal = nStepsTotal
        self.endTime = endTime

        self.nMasses = nMasses
        self.useFriction = useFriction
        
        self.nODE2 = 1 #always 1, as only one is measured / actuated
        self.useVelocities = useVelocities
        self.useInitialValues = useInitialValues
        self.initOnlyLast = initOnlyLast # if True: only initialize the last mass in the chain. No effect for nMasses=1. 
        self.useHarmonicExcitation = useHarmonicExcitation
        self.useRandomExcitation = useRandomExcitation
        self.flagNoForce = flagNoForce
        
        self.nStepForces = nStepForces
        
        self.flagDuffing = flagDuffing # nonlinear spring with cubic term
        self.nonlinearityFactor = nonlinearityFactor # cubic term is spring * nonlinearityfactor * x^3
        
        self.computationType = ModelComputationType.dynamicImplicit

        self.inputStep = False #only constant step functions as input
        self.modelName = 'Non'*(bool(flagDuffing) or bool(useFriction)) + 'linearOscillator' + bool(nMasses>1) * (str(nMasses) + 'Masses')
        self.modelNameShort = 'nl-'*(bool(flagDuffing) or bool(useFriction)) +  'osc '+ bool(nMasses>1) * ('-' + str(nMasses) + 'Mass')
        
        if useInitialValues and nMasses>1:
            print ('Warning: NonlinearOscillator: in case of useInitialValues=True, nMasses should have previously been 1')


        self.variationMKD = variationMKD #add factors for mass, spring, damper (factors 0.5..2)
        self.initHidden = False # True # and self.IsRNN() #for RNN
        self.smallFact = 1. #0.1    #use small values for inputs to remove tanh nonlinearity

        self.nInit = 2*self.nODE2 + 2*(self.nMasses-1) * bool(not(self.initOnlyLast))
        if self.initHidden or not useInitialValues: 
            self.nInit=0
        if self.variationMKD: 
            self.nInit += 3*self.nMasses
        scalForces = 2000.
        self.scalVelocities = 40.
        # self.inputScaling = np.ones((self.nStepsTotal, self.nInit+1+self.variationMKD*3)) #2 initial cond., 1 force, 3*MKD
        self.inputScaling = np.ones((self.nStepsTotal + self.nInit +self.variationMKD*3,1))
        
        # if self.IsFFN():
        #     self.inputScaling = np.ones(2*self.useInitialValues+
        #                                 self.variationMKD*3+
        #                                 self.nStepsTotal #forces
        #                                 ) #2 initial cond., 1 force, 3*MKD
            
        #     # shift column of input force vector by the 3: mass, spring and damper. 
        #     self.inputScaling[self.nInit+self.variationMKD*3:] *= scalForces

        self.inputScaling[self.nInit:] *= scalForces


        self.outputScaling = np.ones((self.nStepsTotal, 1+int(self.useVelocities) )) #displacement + velocity

        if self.useVelocities:
            self.outputScaling[:,1] /= self.scalVelocities # should be normalized for NN

        # else:
        #     self.inputScaling = np.array([1.]*int(self.useInitialValues)*2*self.nODE2 + [scalForces]*self.nStepsTotal)
        #     self.outputScaling = np.array([1.]*self.nStepsTotal + [1.]*(int(self.useVelocities)*self.nStepsTotal ))
        

    def CreateModel(self):
        self.SC = exu.SystemContainer()
        self.mbs = self.SC.AddSystem()
    
        L = 1 #distance of masses, length of springs
        self.mass = 1 #weight of one mass
        #self.nMasses = 1 #number of masses
        self.spring = 1.6e3
        self.damper = 0.005*self.spring
        rMass = 0.1*L
        
        self.resonanceFreq = np.sqrt(self.spring/self.mass)/(2*np.pi) #6.36
        omega = 2*np.pi*self.resonanceFreq
    
        gGround = [GraphicsDataOrthoCubePoint(size=[0.1,0.1,0.1], color=color4grey)]
        oGround = self.mbs.AddObject(ObjectGround(visualization=VObjectGround(graphicsData=gGround)) )

        #ground node for first spring
        nGround=self.mbs.AddNode(NodePointGround(referenceCoordinates = [0,0,0]))
        groundMarker=self.mbs.AddMarker(MarkerNodeCoordinate(nodeNumber= nGround, coordinate = 0))
        prevMarker = groundMarker
        
        if self.flagDuffing and not(self.useFriction): 
             def UFspring(mbs, t, itemNumber, u, v, k, d, F0):
                 return k*u + self.nonlinearityFactor * k * u**3 + d*v
             
        elif self.flagDuffing and self.useFriction: 
            def UFspring(mbs, t, itemNumber, u, v, k, d, F0):
                return k*u + self.nonlinearityFactor * k * u**3+ 50*StribeckFunction(v, muDynamic=1, muStaticOffset=1.5, regVel=1e-4)
             
        elif self.useFriction: 
            def UFspring(mbs, t, itemNumber, u, v, k, d, F0):
                return k*u + 50*StribeckFunction(v, muDynamic=1, muStaticOffset=1.5, regVel=1e-4)


        else:
            UFspring = 0

        gSphere = GraphicsDataSphere(point=[0,0,0], radius=rMass, color=color4blue, nTiles=16)
        lastBody = oGround
        self.massPointList, self.oSDList = [], []
        for i in range(self.nMasses):
            node = self.mbs.AddNode(Node1D(referenceCoordinates = [L*(1+i)],
                                      initialCoordinates=[0.],
                                      initialVelocities=[0.]))
            self.massPoint = self.mbs.AddObject(Mass1D(nodeNumber = node, physicsMass=self.mass,
                                             referencePosition=[0,0,0],
                                             visualization=VMass1D(graphicsData=[gSphere])))

            nodeMarker =self.mbs.AddMarker(MarkerNodeCoordinate(nodeNumber= node, coordinate = 0))
            
            #Spring-Damper between two marker coordinates
            self.oSD = self.mbs.AddObject(CoordinateSpringDamper(markerNumbers = [prevMarker, nodeMarker], 
                                                 stiffness = self.spring, damping = self.damper, 
                                                 springForceUserFunction = UFspring,
                                                 visualization=VCoordinateSpringDamper(drawSize=rMass))) 
            prevMarker = nodeMarker
            self.massPointList += [self.massPoint]
            self.oSDList += [self.oSD]
                
        self.timeVecIn = np.arange(0,self.nStepsTotal)/self.nStepsTotal*self.endTime
        self.timeVecOut = np.arange(1,self.nStepsTotal+1)/self.nStepsTotal*self.endTime
        self.mbs.variables['timeVecOut'] = self.timeVecOut
        
        self.fVec = None
 
        self.mbs.variables['fVec'] = self.fVec
        self.mbs.variables['testForce']= []
        self.mbs.variables['testForce2']= []
        self.mbs.variables['testTime']= []
        self.mbs.variables['h']= self.endTime/self.nStepsTotal
        
        def UFforce(mbs, t, load):
            forceValue = GetInterpolatedSignalValue (t, mbs.variables['fVec'][:,0], mbs.variables['timeVecOut'],
                                               rangeWarning=False)
            iStepNN = int(t / mbs.variables['h'])
            # print('step: ', iStepNN)
            
            if iStepNN == len(mbs.variables['fVec'][:,0]): 
                forceValue2 = mbs.variables['fVec'][-1,0]
            else: 
                forceValue2 = mbs.variables['fVec'][iStepNN,0]
            
            mbs.variables['testForce'] += [forceValue]
            mbs.variables['testTime'] += [t]
            mbs.variables['testForce2'] += [forceValue2]
            
            
            return forceValue2
            
        
        load = self.mbs.AddLoad(LoadCoordinate(markerNumber=prevMarker, load=0, 
                                loadUserFunction=UFforce))
        
        #coordinates of last node are output:
        self.sCoordinates = self.mbs.AddSensor(SensorNode(nodeNumber=node, storeInternal=True,
                                                          outputVariableType=exu.OutputVariableType.Coordinates))
        self.sCoordinates_t = self.mbs.AddSensor(SensorNode(nodeNumber=node, storeInternal=True,
                                                          outputVariableType=exu.OutputVariableType.Coordinates_t))
    
        self.mbs.Assemble()

        self.simulationSettings 
        self.simulationSettings = exu.SimulationSettings() #takes currently set values or default values
        self.simulationSettings.solutionSettings.sensorsWritePeriod = self.endTime / self.nStepsTotal
        self.simulationSettings.timeIntegration.numberOfSteps = self.GetNSimulationSteps()
        self.simulationSettings.timeIntegration.endTime = self.endTime

    #get time vector according to output data
    def GetOutputXAxisVector(self):
        return self.timeVecOut

    #create a randomized input vector
    #relCnt can be used to create different kinds of input vectors (sinoid, noise, ...)
    #isTest is True in case of test data creation
    def CreateInputVector(self, relCnt = 0, isTest=False, dataErrorEstimator = False):
        if dataErrorEstimator: 
            raise ValueError('Todo: implement dataErrorEstimator!')
            
        vec = np.zeros(self.GetInputScaling().shape)
        forces = np.zeros(self.nStepsTotal)
        if self.nStepForces:
            steps = self.smallFact*(2.*np.random.rand(self.nStepForces)-1.) #force values interpolated
            for i, t in enumerate(self.timeVecOut):
                forces[i] = steps[int(self.nStepForces*i/len(self.timeVecOut))]
                
        if self.useHarmonicExcitation:
            omega = 12*2*np.pi*np.random.rand() #6.4 Hz is eigenfrequency of one-mass oscillator
            amp = np.random.rand()
            phi = 2.*np.pi*np.random.rand()
            forces = amp * np.sin(omega*self.timeVecOut+phi) #gives np.array
        if self.useRandomExcitation: 
            forces = np.random.random(forces.shape)

        MKD = []
        if self.variationMKD:
            MKD = np.zeros(3)
            #M,K,D must be all the same for one input vector!
            #these are factors!
            rangeMKD = 4 #4.
            a = 1./rangeMKD
            b = 1-a
            MKD[0] = a+b*np.random.rand() #mass
            MKD[1] = a+b*np.random.rand() #spring
            MKD[2] = a+b*np.random.rand() #damper

        if self.initOnlyLast or self.nMasses == 1: 
            initVals = self.smallFact*0.5*(2*np.random.rand(2)-1.) 
        else: 
            initVals = self.smallFact*0.5*(2*np.random.rand(2 * self.nMasses)-1.) 

        if False and  not self.IsFFN():
            if not self.initHidden and self.useInitialValues:
                vec[:,0] = initVals[0]
                vec[:,1] = initVals[1]
    
            for i, mkd in enumerate(MKD):
                vec[:,(self.nInit+1+i)] = mkd
    
            vec[:,self.nInit] =forces * int(self.flagNoForce)
        else:
            if self.useInitialValues:
                # vec[:,0] = initVals[0:self.nMasses]
                # vec[:,1] = initVals[self.nMasses:self.nMasses*2]
                vec[0:self.nMasses,:] = np.array([initVals[0:self.nMasses]]).T
                vec[self.nMasses:2*self.nMasses,:] = np.array([initVals[self.nMasses:self.nMasses*2]]).T

                # vec[:,0] = initVals[0]
                # vec[:,1] = initVals[1]
            for i, mkd in enumerate(MKD):
                vec[(self.nInit+i)] = mkd
            #print('vec shape', vec.shape, ', force.shape',forces.shape, self.nInit)
            # vec[:,self.nInit+len(MKD)] = forces
            vec[self.nInit+len(MKD):,:] = forces.reshape(vec[self.nInit+len(MKD):,:].shape) * bool(not(int(self.flagNoForce)))
            
        return vec

    #create initialization of (couple of first) hidden states (RNN)
    def CreateHiddenInit(self, isTest):
        if self.initHidden:
            vec = np.zeros(2)
            vec[0] = self.smallFact*0.5*(2*np.random.rand()-1.) #initialODE2
            vec[1] = self.smallFact*0.5*(2*np.random.rand()-1.) #initialODE2_t
            return vec
        else:
            return np.array([])
            
    #get number of simulation steps
    def GetNSimulationSteps(self):
        return self.nStepsTotal*10 # 10 x finer simulation than output

    #split input data into initial values, forces or other inputs
    #return dict with 'data' and possibly 'initialODE2' and 'initialODE2_t'
    def SplitInputData(self, inputData, hiddenData=None):
        data = np.array(self.GetInputScaling()*inputData)
        rv = {}
        if not self.IsFFN():
            if self.initHidden:
                rv['initialODE2'] = [hiddenData[0]]   
                rv['initialODE2_t'] = [hiddenData[1]]#*self.scalVelocities] 
            elif self.useInitialValues:
                if self.initOnlyLast: 
                    if self.nMasses > 1:                     
                        rv['initialODE2'] = np.concatenate((data[0:self.nODE2], (self.nMasses-1)*[[0]]))
                        rv['initialODE2_t'] = np.concatenate((data[self.nODE2:(2*self.nODE2)], (self.nMasses-1)*[[0]])) # data[0,self.nODE2:(2*self.nODE2)]#
                    else: 
                        rv['initialODE2'] = data[0:self.nODE2]
                        rv['initialODE2_t'] = data[self.nODE2:(2*self.nODE2)] 
                else:
                    rv['initialODE2'] = data[0:self.nODE2*self.nMasses]
                    rv['initialODE2_t'] = data[self.nODE2*self.nMasses:(2*self.nODE2*self.nMasses)] # data[0,self.nODE2:(2*self.nODE2)]


            # iOffset = 0
            if self.variationMKD:
                rv['MKD'] = data[(self.nInit+1):(self.nInit+4),:] #MKD are the same for all sequences
                # iOffset = 3
            rv['data'] = data[self.nInit:,:] #forces
        else:
            off = 0
            if self.useInitialValues:
                rv['initialODE2'] = data[0:self.nODE2]
                rv['initialODE2_t'] = data[self.nODE2:(2*self.nODE2)]
                off+=2
    
            if self.variationMKD:
                rv['MKD'] = data[off:(off+3)] #MKD are the same for all sequences
                off+=3

            rv['data'] = data[off:] #forces
        
        return rv

    #split output data to get ODE2 values (and possibly other data, such as ODE2)
    #return dict {'ODE2':[], 'ODE2_t':[]}
    def SplitOutputData(self, outputData):
        rv = {}
        data = outputData
        if outputData.ndim == 1:
            data = outputData.reshape((self.nStepsTotal,1+self.useVelocities))
        rv['ODE2'] = data[:,0]
        if self.useVelocities:
            rv['ODE2_t'] = data[:,1]
        return rv
    
    #convert all output vectors into plottable data (e.g. [time, x, y])
    #the size of data allows to decide how many columns exist
    def OutputData2PlotData(self, outputData, forSolutionViewer=False):
        if self.nMasses != 1 and forSolutionViewer:
            raise ValueError('NonlinearOscillator.OutputData2PlotData: nMasses > 1 is not suitable for SolutionViewer!')
        timeVec = self.GetOutputXAxisVector()
        dataDict = self.SplitOutputData(outputData)
        
        if 'ODE2_t' in dataDict and not forSolutionViewer:
            data = np.vstack((timeVec, dataDict['ODE2'].T, dataDict['ODE2_t'].T)).T
        else:
            data = np.vstack((timeVec, dataDict['ODE2'].T)).T
            
        return data

    #return dict of names to columns for plotdata        
    def PlotDataColumns(self):
        d = {'time':0, 'ODE2':1}
        if self.useVelocities:
            d['ODE2_t'] = 2
        
        return d

    #get compute model with given input data and return output data
    #initialState contains position and velocity states as list of two np.arrays 
    def ComputeModel(self, inputData, hiddenData=None, verboseMode = 0, solutionViewer = False):
        #set input data ...
        inputDict = self.SplitInputData(np.array(inputData), hiddenData)

        if self.variationMKD:
            #mass enters the equations reciprocal ...
            for (massPoint, oSD) in zip(self.massPointList, self.oSDList): 
                self.mbs.SetObjectParameter(massPoint, 'physicsMass', self.mass/(2*inputDict['MKD'][0]))
                self.mbs.SetObjectParameter(oSD, 'stiffness', self.spring*2*inputDict['MKD'][1])
                self.mbs.SetObjectParameter(oSD, 'damping', self.damper*1*inputDict['MKD'][2])
            
            #self.CreateModel() #must be created newly for each test ...


        # inputDict = self.SplitInputData(self.GetInputScaling() * np.array(inputData))

        if 'initialODE2' in inputDict:
            if  self.nMasses == 1: 
                #print('initODE2',inputDict['initialODE2'])
                self.mbs.systemData.SetODE2Coordinates(inputDict['initialODE2'], configuration=exu.ConfigurationType.Initial)
                self.mbs.systemData.SetODE2Coordinates_t(inputDict['initialODE2_t'], configuration=exu.ConfigurationType.Initial)
            else: 
                self.mbs.systemData.SetODE2Coordinates( inputDict['initialODE2'], configuration=exu.ConfigurationType.Initial)
                self.mbs.systemData.SetODE2Coordinates_t(inputDict['initialODE2_t'], configuration=exu.ConfigurationType.Initial)
                
        self.mbs.variables['fVec'] = inputDict['data']

        self.simulationSettings.timeIntegration.verboseMode = verboseMode
        self.simulationSettings.solutionSettings.writeSolutionToFile = solutionViewer 
 
        solverType = exu.DynamicSolverType.TrapezoidalIndex2
        if self.useFriction:
            solverType = exu.DynamicSolverType.ExplicitMidpoint
        #solverType = exu.DynamicSolverType.ExplicitEuler
        
        self.mbs.SolveDynamic(self.simulationSettings, 
                         solverType = solverType)
        if solutionViewer:
            self.mbs.SolutionViewer()

        #get sensor data and apply piecewise scaling:
        output = 0*self.GetOutputScaling()
        output[:,0] = self.mbs.GetSensorStoredData(self.sCoordinates)[1:,1] #sensordata includes time
        if self.useVelocities:
            output[:,1] = self.mbs.GetSensorStoredData(self.sCoordinates_t)[1:,1] #sensordata includes time
        
        output = self.GetOutputScaling()*output
        if False: 
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2)
            axs[0].plot(self.timeVecIn, inputDict['data'], 'x')
            axs[0].plot(self.mbs.variables['testTime'], self.mbs.variables['testForce2'])
            axs[1].plot(self.timeVecOut, output)
            
        return output





#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class NonlinearBeamStatic(SimulationModel):

    #initialize class 
    def __init__(self, nBeams=16):
        SimulationModel.__init__(self)

        self.nBeams = nBeams
        self.nNodes = self.nBeams+1
        self.nODE2 = (self.nNodes)*4
        self.computationType = ModelComputationType.static

        self.modelName = 'NonlinearBeamStatic'
        self.modelNameShort = 'nl-beam'
        
        scalForces = 2.
        self.inputScaling = np.array([scalForces,scalForces]) #Fx, Fy
        self.outputScaling = np.array([1.]*(self.nNodes*2) ) #x/y positions of nodes
        

    def CreateModel(self):
        self.SC = exu.SystemContainer()
        self.mbs = self.SC.AddSystem()
    
        L = 1 #total beam length
            
        E=2.07e9                # Young's modulus of ANCF element in N/m^2
        rho=7800               # density of ANCF element in kg/m^3
        b=0.001                # width of rectangular ANCF element in m
        h=0.002                # height of rectangular ANCF element in m
        A=b*h                  # cross sectional area of ANCF element in m^2
        I=b*h**3/12            # second moment of area of ANCF element in m^4
        
        #generate ANCF beams with utilities function
        cableTemplate = Cable2D(#physicsLength = L / nElements, #set in GenerateStraightLineANCFCable2D(...)
                                physicsMassPerLength = rho*A,
                                physicsBendingStiffness = E*I,
                                physicsAxialStiffness = E*A,
                                physicsBendingDamping=E*I*0.05,
                                useReducedOrderIntegration = 1,
                                #nodeNumbers = [0, 0], #will be filled in GenerateStraightLineANCFCable2D(...)
                                )
        
        positionOfNode0 = [0, 0, 0] # starting point of line
        positionOfNode1 = [L, 0, 0] # end point of line
        
        self.xAxis = np.arange(0,L,L/(self.nBeams+1))
        
        #alternative to mbs.AddObject(Cable2D(...)) with nodes:
        ancf=GenerateStraightLineANCFCable2D(self.mbs,
                        positionOfNode0, positionOfNode1,
                        self.nBeams,
                        cableTemplate, #this defines the beam element properties
                        #massProportionalLoad = [0,-9.81*0,0], #optionally add gravity
                        fixedConstraintsNode0 = [1,1,0,1], #add constraints for pos and rot (r'_y)
                        fixedConstraintsNode1 = [0,0,0,0])
        mANCFLast = self.mbs.AddMarker(MarkerNodePosition(nodeNumber=ancf[0][-1])) #ancf[0][-1] = last node
        
        self.lTipLoad = self.mbs.AddLoad(Force(markerNumber = mANCFLast, 
                                         loadVector = [0, 0, 0], )) 
            
        self.listPosSensors = []
        for node in ancf[0]:
            sPos = self.mbs.AddSensor(SensorNode(nodeNumber=node, storeInternal=True,
                                                 outputVariableType=exu.OutputVariableType.Position))
            self.listPosSensors += [sPos]

    
        self.mbs.Assemble()

        self.simulationSettings 
        self.simulationSettings = exu.SimulationSettings() #takes currently set values or default values
        self.simulationSettings.staticSolver.numberOfLoadSteps = 20
        #reduce tolerances to achieve convergence also for small loads
        self.simulationSettings.staticSolver.newton.absoluteTolerance = 1e-6
        self.simulationSettings.staticSolver.newton.relativeTolerance = 1e-6

    #get time vector according to output data
    def GetOutputXAxisVector(self):
        return self.xAxis

    #create a randomized input vector
    #relCnt can be used to create different kinds of input vectors (sinoid, noise, ...)
    #isTest is True in case of test data creation
    def CreateInputVector(self, relCnt = 0, isTest=False, dataErrorEstimator = False):
        if dataErrorEstimator: 
            raise ValueError('Todo: implement dataErrorEstimator!')
        vec = 2.*np.random.rand(*self.GetInputScaling().shape)-1.

        return vec
            
    #split input data into initial values, forces or other inputs
    #return dict with 'data' and possibly 'initialODE2' and 'initialODE2_t'
    def SplitInputData(self, inputData, hiddenData):
        data = np.array(inputData)
        rv = {}
        rv['data'] = data
        return rv

    #split output data to get ODE2 values (and possibly other data, such as ODE2)
    #return dict {'ODE2':[], 'ODE2_t':[]}
    def SplitOutputData(self, outputData):
        rv = {}
        rv['ODE2_x'] = outputData[0:self.nNodes]
        rv['ODE2_y'] = outputData[self.nNodes:]
        return rv
    
    #convert all output vectors into plottable data (e.g. [time, x, y])
    #the size of data allows to decide how many columns exist
    def OutputData2PlotData(self, outputData, forSolutionViewer=False):
        if forSolutionViewer:
            raise ValueError('NonlinearBeamStatic.OutputData2PlotData: this model is not suitable for SolutionViewer!')
        dataDict = self.SplitOutputData(outputData)
        
        data = np.vstack((self.GetOutputXAxisVector(), dataDict['ODE2_x'].T, dataDict['ODE2_y'].T)).T
        return data

    #return dict of names to columns for plotdata        
    def PlotDataColumns(self):
        return {'x':0, 'ODE2_x':1, 'ODE2_y':2}

    #get compute model with given input data and return output data
    #initialState contains position and velocity states as list of two np.arrays 
    def ComputeModel(self, inputData, hiddenData=None, verboseMode = 0, solutionViewer = False):
        #set input data ...
        inputDict = self.SplitInputData(self.GetInputScaling() * np.array(inputData))
        loadVector = list(inputDict['data'])+[0]
        
        self.mbs.SetLoadParameter(self.lTipLoad, 'loadVector', loadVector)

        self.simulationSettings.staticSolver.verboseMode = verboseMode
        self.simulationSettings.solutionSettings.writeSolutionToFile = solutionViewer 
 
 
        self.mbs.SolveStatic(self.simulationSettings)
        if solutionViewer:
            print('load=',loadVector)
            self.mbs.SolutionViewer()

        #get sensor data and apply piecewise scaling:
        output = 0*self.GetOutputScaling()
        for i, sPos in enumerate(self.listPosSensors):
            p = self.mbs.GetSensorValues(sPos)
            output[i] = p[0] #x
            output[i+self.nNodes] = p[1] #y

        output = self.GetOutputScaling()*output
        return output

    def SolutionViewer(self, outputData, outputDataColumns = [0]):
        print('NonlinearBeamStatic.SolutionViewer: not available')




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#follows Multibody System Dynamics 2021 paper of Choi et al. "Data-driven simulation for general-purpose multibody dynamics using Deep Neural Networks"
#NOTE: wrong theta0 in paper!
class DoublePendulum(SimulationModel):

    #initialize class 
    def __init__(self, nStepsTotal=100, useInitialVelocities=True, useInitialAngles=True, 
                 addTime=False, endTime=1, variationLengths=True):
        SimulationModel.__init__(self)

        #required in base:
        self.nStepsTotal = nStepsTotal
        self.useInitialVelocities = useInitialVelocities
        self.useInitialAngles = useInitialAngles
        self.endTime = endTime
        self.variationLengths = variationLengths

        self.smallFact = 1 #0.1    #use small values for inputs to remove tanh nonlinearity
        self.addTime = addTime #add time as input

        self.nODE2 = 2 
        self.m0 = 2 #kg
        self.m1 = 1 #kg
        self.L0 = 1 #m
        self.L1 = 2 #m
        self.gravity = 9.81 #m/s**2
        # self.initAngles = np.array([0.5*pi,0.5*pi]) #initial angle from vertical equilibrium
        self.initAngles = np.array([1.6,2.2]) #initial angle ranges from vertical equilibrium
        self.initAngles_t = 0*np.array([0.1,0.5]) #initial angular velocity ranges

        #default values, if not varied:
        self.phi0 = 1.6 #self.smallFact*0.5*pi #will be set later
        self.phi1 = 2.2 #self.smallFact*0.5*pi #will be set later; WRONG in paper!
        self.phi0_t = 0 #will be set later
        self.phi1_t = 0 #will be set later
        
        self.computationType = ModelComputationType.dynamicImplicit

        self.modelName = 'DoublePendulum'
        self.modelNameShort = 'd-pend'
        
        self.initHidden = True and self.IsRNN() #for RNN

        #for FFN, we can test also with no initial angles; for RNN it is always needed
        #nInit: 0..1=angle0/1, 2..3=vel0/1
        self.nInit = self.nODE2*(int(useInitialVelocities) + int(useInitialAngles))
        if self.initHidden: 
            self.nInit=0
        
        self.scalVelocities = 5.
        #inputscaling is difficult for RNN (hidden states), better not to use
        if self.IsFFN():
            self.inputScaling = np.ones(self.nInit + self.variationLengths*2) 
        else:
            self.inputScaling = np.ones((self.nStepsTotal, 2*int(self.variationLengths) + 
                                         int(self.addTime))) #2 lengths + time


        self.outputScaling = np.ones((self.nStepsTotal, 2*self.nODE2 )) #2*displacements + 2*velocities)
        self.outputScaling[:,2] /= self.scalVelocities
        self.outputScaling[:,3] /= self.scalVelocities

        

    def CreateModel(self):
        self.SC = exu.SystemContainer()
        self.mbs = self.SC.AddSystem()
    
        
    
        b = 0.1
        gGround = [GraphicsDataOrthoCubePoint(centerPoint=[0,0.5*b,0],size=[b,b,b], color=color4grey)]
        oGround = self.mbs.AddObject(ObjectGround(visualization=VObjectGround(graphicsData=gGround)) )

        #put double pendulum in 
        refP0 = [0.,-self.L0,0.]
        p0 = np.array([self.L0*sin(self.phi0),-self.L0*cos(self.phi0),0.])
        omega0 = [0.,0.,self.phi0_t]
        v0 = np.cross(omega0, p0)
        o0 = self.mbs.CreateMassPoint(referencePosition=refP0,
                                      initialDisplacement=p0-refP0,
                                      initialVelocity=v0,
                                      physicsMass=self.m0,
                                      gravity=[0.,-self.gravity,0.],
                                      create2D=True,
                                      drawSize = b,color=color4red)

        refP1 = [0.,-self.L0-self.L1,0.]
        p1 = p0 + [self.L1*sin(self.phi1),-self.L1*cos(self.phi1),0.]
        omega1 = [0.,0.,self.phi1_t]
        v1 = np.cross(omega1, p1-p0)+v0
        o1 = self.mbs.CreateMassPoint(referencePosition=refP1,
                                      initialDisplacement=p1-refP1,
                                      initialVelocity=v1,
                                      physicsMass=self.m1,
                                      gravity=[0.,-self.gravity,0.],
                                      create2D=True,
                                      drawSize = b,color=color4dodgerblue)
        #print('p0=',p0, ', p1=', p1)
        
        self.mbs.CreateDistanceConstraint(bodyOrNodeList=[oGround,o0], distance=self.L0)
        self.mbs.CreateDistanceConstraint(bodyOrNodeList=[o0,o1], distance=self.L1)
        
        self.sPos0 = self.mbs.AddSensor(SensorBody(bodyNumber=o0, storeInternal=True,
                                              outputVariableType=exu.OutputVariableType.Position))
        self.sPos1 = self.mbs.AddSensor(SensorBody(bodyNumber=o1, storeInternal=True,
                                              outputVariableType=exu.OutputVariableType.Position))
        self.sVel0 = self.mbs.AddSensor(SensorBody(bodyNumber=o0, storeInternal=True,
                                              outputVariableType=exu.OutputVariableType.Velocity))
        self.sVel1 = self.mbs.AddSensor(SensorBody(bodyNumber=o1, storeInternal=True,
                                              outputVariableType=exu.OutputVariableType.Velocity))
        
        def UFsensor(mbs, t, sensorNumbers, factors, configuration):
            p0 = mbs.GetSensorValues(sensorNumbers[0]) 
            p1 = mbs.GetSensorValues(sensorNumbers[1]) 
            v0 = mbs.GetSensorValues(sensorNumbers[2]) 
            v1 = mbs.GetSensorValues(sensorNumbers[3]) 
            phi0 = atan2(p0[0],-p0[1]) #compute angle; straight down is zero degree
            dp1 = p1-p0 #relative position
            dv1 = v1-v0 #relative velocity
            phi1 = atan2(dp1[0],-dp1[1]) 
            
            nom0 = p0[0]**2+p0[1]**2
            phi0_t = -p0[1]/nom0 * v0[0] + p0[0]/nom0 * v0[1]

            nom1 = dp1[0]**2+dp1[1]**2
            phi1_t = -dp1[1]/nom1 * dv1[0] + dp1[0]/nom1 * dv1[1]
            
            return [phi0,phi1,phi0_t,phi1_t] 

        self.sAngles = self.mbs.AddSensor(SensorUserFunction(sensorNumbers=[self.sPos0,self.sPos1,self.sVel0,self.sVel1], 
                                                             #factors=[self.L0, self.L1],
                                 storeInternal=True,sensorUserFunction=UFsensor))
        
    
        self.timeVecOut = np.arange(1,self.nStepsTotal+1)/self.nStepsTotal*self.endTime
    
        self.mbs.Assemble()

        self.simulationSettings 
        self.simulationSettings = exu.SimulationSettings() #takes currently set values or default values
        self.simulationSettings.solutionSettings.sensorsWritePeriod = self.endTime / self.nStepsTotal
        self.simulationSettings.timeIntegration.numberOfSteps = self.GetNSimulationSteps()
        self.simulationSettings.timeIntegration.endTime = self.endTime

    #get time vector according to output data
    def GetOutputXAxisVector(self):
        return self.timeVecOut

    #create a randomized input vector
    #relCnt can be used to create different kinds of input vectors (sinoid, noise, ...)
    #isTest is True in case of test data creation
    def CreateInputVector(self, relCnt = 0, isTest=False, dataErrorEstimator = False):
        vec = np.zeros(self.GetInputScaling().shape)
        #print(vec.shape)
        lengths = []
        if self.variationLengths:
            lengths = np.zeros(2)

            lengths[0] = 1.+1.*np.random.rand() #L0
            lengths[1] = 2.+1.*np.random.rand() #L1

        randAngles = self.smallFact*(self.initAngles*(np.random.rand(2)))
        randVels = self.smallFact*self.initAngles_t*np.random.rand(2)

        if not self.useInitialAngles:
            randAngles = self.initAngles
        if not self.useInitialVelocities:
            randVels = [0.,0.]

        if not self.IsFFN():
            for i, length in enumerate(lengths):
                vec[:,i] = length
            if self.addTime:
                vec[:,0+len(lengths)] = self.timeVecOut
        else:
            off = 0
            if self.useInitialAngles:
                vec[0] = randAngles[0]
                vec[1] = randAngles[1]
                off += 2
            if self.useInitialVelocities:
                vec[0+off] = randVels[0]
                vec[1+off] = randVels[1]
                off += 2
    
            for i, length in enumerate(lengths):
                vec[off+i] = length
            off += len(lengths)

            # if self.addTime: # attaching output time vector did not improve results
            #     vec[off:] = self.timeVecOut
            
        return vec

    #create initialization of (couple of first) hidden states (RNN)
    def CreateHiddenInit(self, isTest):
        if self.initHidden:
            vec = np.zeros(2*self.nODE2)
            randAngles = self.smallFact*self.initAngles * (2*np.random.rand(self.nODE2)-1.) 
            randVels = self.smallFact*self.initAngles_t*np.random.rand(self.nODE2)
            
            vec[0:2] = randAngles
            vec[2:4] = randVels

            return vec
        else:
            return np.array([])
            
    #get number of simulation steps
    def GetNSimulationSteps(self):
        return self.nStepsTotal*10 #10 x finer simulation than output

    #split input data into initial values, forces or other inputs
    #return dict with 'data' and possibly 'initialODE2' and 'initialODE2_t'
    def SplitInputData(self, inputData, hiddenData=None):
        data = np.array(self.GetInputScaling()*inputData)
        rv = {}
        if not self.IsFFN():
            if self.initHidden: #always true for RNN
                rv['phi0'] = hiddenData[0]  
                rv['phi1'] = hiddenData[1]  
                rv['phi0_t'] = hiddenData[2]
                rv['phi1_t'] = hiddenData[3]
    
            if self.variationLengths:
                rv['L0'] = data[0,0] #lengths are the same for all sequences
                rv['L1'] = data[0,1] #lengths are the same for all sequences
            if self.addTime: #not needed ...
                rv['time'] = data[:,2*int(self.variationLengths)] #lengths are the same for all sequences
                
        else:
            off = 0
            #default values, if not otherwise set
            if self.useInitialAngles:
                rv['phi0'] = data[0]
                rv['phi1'] = data[1]
                off+=2
            if self.useInitialVelocities:
                rv['phi0_t'] = data[off+0]
                rv['phi1_t'] = data[off+1]
                off+=2
    
            if self.variationLengths:
                rv['L0'] = data[off+0] 
                rv['L1'] = data[off+1] 
                off+=2

            # if self.addTime:
            #     rv['time'] = data[off:] #lengths are the same for all sequences
        
        return rv

    #split output data to get ODE2 values (and possibly other data, such as ODE2)
    #return dict {'ODE2':[], 'ODE2_t':[]}
    def SplitOutputData(self, outputData):
        rv = {}
        data = outputData
        if outputData.ndim == 1:
            data = outputData.reshape((self.nStepsTotal,4))
        rv['phi0'] = data[:,0]
        rv['phi1'] = data[:,1]
        rv['phi0_t'] = data[:,2]
        rv['phi1_t'] = data[:,3]
        return rv
    
    #convert all output vectors into plottable data (e.g. [time, x, y])
    #the size of data allows to decide how many columns exist
    def OutputData2PlotData(self, outputData, forSolutionViewer=False):
        timeVec = self.GetOutputXAxisVector()
        dataDict = self.SplitOutputData(outputData)
        
        data = np.vstack((timeVec, dataDict['phi0'].T, dataDict['phi1'].T,
                          dataDict['phi0_t'].T, dataDict['phi1_t'].T)).T
            
        return data

    #return dict of names to columns for plotdata        
    def PlotDataColumns(self):
        d = {'time':0, 'phi0':1, 'phi1':2, 'phi0_t':3, 'phi1_t':4}
        
        return d

    #get compute model with given input data and return output data
    #initialState contains position and velocity states as list of two np.arrays 
    def ComputeModel(self, inputData, hiddenData=None, verboseMode = 0, solutionViewer = False):
        #set input data ...
        inputDict = self.SplitInputData(np.array(inputData), hiddenData)

        #print('hiddenData=', hiddenData)
        if 'L0' in inputDict:
            self.L0 = inputDict['L0']
            self.L1 = inputDict['L1']
        
        if 'phi0' in inputDict:
            self.phi0 = inputDict['phi0']
            self.phi1 = inputDict['phi1']
        if 'phi0_t' in inputDict:
            self.phi0_t = inputDict['phi0_t']
            self.phi1_t = inputDict['phi1_t']
                    
        self.CreateModel() #must be created newly for each test ...


        self.simulationSettings.timeIntegration.verboseMode = verboseMode
        self.simulationSettings.solutionSettings.writeSolutionToFile = solutionViewer 
 
        self.mbs.SolveDynamic(self.simulationSettings) #GeneralizedAlpha

        if solutionViewer:
            self.mbs.SolutionViewer()

        #get sensor data and apply piecewise scaling:
        output = 0*self.GetOutputScaling()
        
        for i in range(4): #2 x phi and 2 x phi_t 
            #sensordata includes time
            #exclude t=0
            output[:,i] = self.mbs.GetSensorStoredData(self.sAngles)[1:,1+i] 
        
        output = self.GetOutputScaling()*output
        return output

    def SolutionViewer(self, outputData, outputDataColumns = [0]):
        #model is 2D!
        nColumns = 2*2 #2 x (x,y)
        angles = self.OutputData2PlotData(outputData, forSolutionViewer=True)
        #print(angles)
        data = np.zeros((self.nStepsTotal, 1+nColumns))
        for i, t in enumerate(self.timeVecOut):
            data[i,0] = t
            data[i,1] = self.L0*sin(angles[i,1])
            data[i,2] = +self.L0*(1-cos(angles[i,1]))
            data[i,3] = data[i,1]+self.L1*sin(angles[i,2])
            data[i,4] = data[i,2]-self.L1*cos(angles[i,2])+self.L1
        
        # print(data)
        
        # columnsExported = dict({'nODE2':self.nODE2, 
        #                         'nVel2':0, 'nAcc2':0, 'nODE1':0, 'nVel1':0, 'nAlgebraic':0, 'nData':0})
        columnsExported = [nColumns, 0, 0, 0, 0, 0, 0] #nODE2 without time
        if data.shape[1]-1 != nColumns:
            raise ValueError('SimulationModel.SolutionViewer: problem with shape of data: '+
                             str(nColumns)+','+str(data.shape))

        nRows = data.shape[0]
        
        
        sol = dict({'data': data, 'columnsExported': columnsExported,'nColumns': nColumns,'nRows': nRows})

        self.mbs.SolutionViewer(sol,runOnStart=True)





#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# this simulation model does not have equal inout and output step lengths
class NonlinearOscillatorContinuous(SimulationModel):
    #initialize class 
    def __init__(self, nStepsTotal=100,
                 useInitialValues=True, 
                 frictionForce=0., #mu*F_N acting as dry friction force on oscillator
                 nMasses=1, endTime=1, nnType='FFN',
                 scalBaseMotion = 0.2, nonlinearityFactor = 0., #factor for cubic nonlinearity
                 initUfact = 1., initVfact = 10., #factors for randomization of initial values
                 nOutputSteps = None, useVelocities = False, 
                 useFriction = False, useRandomExcitation = True, useHarmonicExcitation = False, 
                 ):
        SimulationModel.__init__(self)

        #required in base:
        self.nStepsTotal = nStepsTotal
        # self.nnType = nnType
        self.endTime = endTime
        self.nOutputSteps = nOutputSteps
        if nOutputSteps == None or nOutputSteps > nStepsTotal:
            self.nOutputSteps = nStepsTotal

        self.initUfact = initUfact
        self.initVfact = initVfact

        self.nMasses = nMasses
        self.frictionForce = frictionForce
        
        self.useHarmonicExcitation = useHarmonicExcitation
        self.useRandomExcitation = useRandomExcitation
        
        self.nonlinearityFactor = nonlinearityFactor
        if self.nonlinearityFactor != 0: 
            self.flagDuffing = True
            
        self.nODE2 = 1 #always 1, as only one is measured / actuated
        self.useVelocities = useVelocities
        if useVelocities: 
            print('warning: useVelocities is True but not implemented properly')
        self.useInitialValues = useInitialValues
        
        self.computationType = ModelComputationType.dynamicImplicit

        self.modelName = 'OSCIcontinuous'
        self.modelNameShort = 'osc-cont'
        
        if useInitialValues and nMasses>1:
            raise ValueError('NonlinearOscillatorContinuous: in case of useInitialValues=True, nMasses must be 1')


        self.nInit = 2*self.nODE2
        if not useInitialValues: 
            self.nInit=0
        
        self.scalBaseMotion = scalBaseMotion
        
        self.inputScaling = np.ones((self.nStepsTotal, #base motion
                                    1))
        self.inputScaling *= scalBaseMotion

        self.outputScaling = np.ones((self.nOutputSteps, 1 )) #mass displacements

        

    def CreateModel(self):
        self.SC = exu.SystemContainer()
        self.mbs = self.SC.AddSystem()
    
        L = 1 #distance of masses, length of springs
        self.mass = 1 #weight of one mass
        self.spring = 1.6e3
        self.damper = 0.005*self.spring # 8
        rMass = 0.1*L # visualization only
        
        self.resonanceFreq = np.sqrt(self.spring/self.mass)/(2*np.pi) #6.36
        omega = 2*np.pi*self.resonanceFreq
    
        gGround = [GraphicsDataOrthoCubePoint(size=[0.1,0.1,0.1], color=color4grey)]
        oGround = self.mbs.AddObject(ObjectGround(visualization=VObjectGround(graphicsData=gGround)) )

        #ground node for first spring
        nGround=self.mbs.AddNode(NodePointGround(referenceCoordinates = [0,0,0]))
        groundMarker=self.mbs.AddMarker(MarkerNodeCoordinate(nodeNumber= nGround, coordinate = 0))
        prevMarker = groundMarker
        
        self.mbs.variables['nonlinearityFactor'] = self.nonlinearityFactor
            
        def UFspring(mbs, t, itemNumber, u, v, k, d, F0):
            off = GetInterpolatedSignalValue (t, mbs.variables['uVec'], mbs.variables['timeVecOut'],
                                           rangeWarning=False)
            # print('t: ', t, 'timeoutVec max: ',  mbs.variables['timeVecOut'][-1])
            Ff = 0
            if self.frictionForce != 0.:
                Ff = self.frictionForce * StribeckFunction(v, muDynamic=1, muStaticOffset=1.5, regVel=1e-4)
            return k*(u-off) + d*v + k*mbs.variables['nonlinearityFactor']*(u-off)**3 + Ff

        gSphere = GraphicsDataSphere(point=[0,0,0], radius=rMass, color=color4blue, nTiles=16)
        lastBody = oGround
        for i in range(self.nMasses):
            node = self.mbs.AddNode(Node1D(referenceCoordinates = [L*(1+i)],
                                      initialCoordinates=[0.],
                                      initialVelocities=[0.]))
            self.massPoint = self.mbs.AddObject(Mass1D(nodeNumber = node, physicsMass=self.mass,
                                             referencePosition=[0,0,0],
                                             visualization=VMass1D(graphicsData=[gSphere])))

            nodeMarker =self.mbs.AddMarker(MarkerNodeCoordinate(nodeNumber= node, coordinate = 0))
            
            #Spring-Damper between two marker coordinates
            self.oSD = self.mbs.AddObject(CoordinateSpringDamper(markerNumbers = [prevMarker, nodeMarker], 
                                                 stiffness = self.spring, damping = self.damper, 
                                                 springForceUserFunction = UFspring,
                                                 visualization=VCoordinateSpringDamper(drawSize=rMass))) 
            prevMarker = nodeMarker
                
        self.timeVecIn = np.arange(0,self.nStepsTotal)/self.nStepsTotal*self.endTime
        self.timeVecOut = np.arange(1,self.nStepsTotal+1)/self.nStepsTotal*self.endTime
        self.mbs.variables['timeVecOut'] = self.timeVecOut
        
        self.uVec = None
 
        self.mbs.variables['uVec'] = np.zeros(self.mbs.variables['timeVecOut'].shape)
        

            
        
        # load = self.mbs.AddLoad(LoadCoordinate(markerNumber=prevMarker, load=0, 
        #                         loadUserFunction=UFforce))
        
        #coordinates of last node are output:
        self.sCoordinates = self.mbs.AddSensor(SensorNode(nodeNumber=node, storeInternal=True,
                                                          outputVariableType=exu.OutputVariableType.Coordinates))
        self.sCoordinates_t = self.mbs.AddSensor(SensorNode(nodeNumber=node, storeInternal=True,
                                                          outputVariableType=exu.OutputVariableType.Coordinates_t))
    
        self.mbs.Assemble()

        self.simulationSettings 
        self.simulationSettings = exu.SimulationSettings() #takes currently set values or default values
        self.simulationSettings.solutionSettings.sensorsWritePeriod = self.endTime / self.nStepsTotal
        self.simulationSettings.timeIntegration.numberOfSteps = self.GetNSimulationSteps()
        self.simulationSettings.timeIntegration.endTime = self.endTime

    #get time vector according to output data
    def GetOutputXAxisVector(self):
        return self.timeVecOut

    #create a randomized input vector
    #relCnt can be used to create different kinds of input vectors (sinoid, noise, ...)
    #isTest is True in case of test data creation
    def CreateInputVector(self, relCnt = 0, isTest=False, dataErrorEstimator=False):
        scaleInput = 1
        if dataErrorEstimator: 
            scaleInput = 3
            # raise ValueError('Todo: implement dataErrorEstimator!')
        
            
        vec = np.zeros(self.GetInputScaling().shape)
        baseMotion = np.zeros(self.nStepsTotal)
        flagRandomExc = self.useRandomExcitation
        flagHarmonicExc = self.useHarmonicExcitation
        # if both flags are true, split 50/50!
        if flagRandomExc and flagHarmonicExc: 
            if np.random.random() > 0.5: 
                flagRandomExc = False 

        if flagHarmonicExc: 
            w = np.random.uniform(5, 20)
            phi = np.random.uniform(-np.pi, np.pi)
        vLast = 0
        for i, t in enumerate(self.timeVecOut):
            if flagRandomExc: 
                v = (np.random.rand()*2-1) #range +1/-1
                baseMotion[i] = v*0.1 + vLast*0.9
                vLast = v
            elif flagHarmonicExc: 
                # v = []
                baseMotion[i] = np.sin(t*w + phi)
            
    
        vec[:,0] = baseMotion
        # if dataErrorEstimator: 
        vec[:,0] = vec[:,0] * scaleInput
            # print('')
        return vec

    #create initialization of (couple of first) hidden states (RNN)
    def CreateHiddenInit(self, isTest):
        return np.array([])
            
    #get number of simulation steps
    def GetNSimulationSteps(self):
        return self.nStepsTotal*10 #10 x finer simulation than output

    #split input data into initial values, forces or other inputs
    #return dict with 'data' and possibly 'initialODE2' and 'initialODE2_t'
    def SplitInputData(self, inputData, hiddenData=None):
        data = np.array(self.GetInputScaling()*inputData)
        rv = {}

        rv['data'] = data #baseMotion
        
        return rv

    #split output data to get ODE2 values (and possibly other data, such as ODE2)
    #return dict {'ODE2':[], 'ODE2_t':[]}
    def SplitOutputData(self, outputData):
        rv = {}
        data = outputData
        if outputData.ndim == 1:
            data = outputData.reshape((self.nOutputSteps,1))
        rv['ODE2'] = data[:,0]
        return rv
    
    #convert all output vectors into plottable data (e.g. [time, x, y])
    #the size of data allows to decide how many columns exist
    def OutputData2PlotData(self, outputData, forSolutionViewer=False):
        if self.nMasses != 1 and forSolutionViewer:
            raise ValueError('NonlinearOscillator.OutputData2PlotData: nMasses > 1 is not suitable for SolutionViewer!')
        
        nDiff = self.nStepsTotal - self.nOutputSteps
        leadingZeros = np.zeros(nDiff)
        dataDict = self.SplitOutputData(outputData)

        #timeVec = np.hstack((leadingZeros, self.GetOutputXAxisVector() ))
        timeVec = self.GetOutputXAxisVector()
        ode2 = np.hstack((leadingZeros, dataDict['ODE2'] ))
        
        data = np.vstack((timeVec, ode2.T)).T
            
        return data

    #return dict of names to columns for plotdata        
    def PlotDataColumns(self):
        d = {'time':0, 'ODE2':1}
        
        return d

    #get compute model with given input data and return output data
    #initialState contains position and velocity states as list of two np.arrays 
    def ComputeModel(self, inputData, hiddenData=None, verboseMode = 0, solutionViewer = False):
        #set input data ...
        inputDict = self.SplitInputData(np.array(inputData), hiddenData)

        #we create random initial values, which we do not know!
        initU = self.initUfact*(np.random.rand(self.nODE2)*2-1)
        initV = self.initVfact*(np.random.rand(self.nODE2)*2-1)
        
        self.mbs.systemData.SetODE2Coordinates(initU, configuration=exu.ConfigurationType.Initial)
        self.mbs.systemData.SetODE2Coordinates_t(initV, configuration=exu.ConfigurationType.Initial)

        self.mbs.variables['uVec'] = inputDict['data'][:,0]

        self.simulationSettings.timeIntegration.verboseMode = verboseMode
        self.simulationSettings.solutionSettings.writeSolutionToFile = solutionViewer 
 
        # solverType = exu.DynamicSolverType.TrapezoidalIndex2
        # if self.frictionForce != 0.:
        solverType = exu.DynamicSolverType.ExplicitMidpoint
        #solverType = exu.DynamicSolverType.ExplicitEuler
        
        self.mbs.SolveDynamic(self.simulationSettings, 
                         solverType = solverType)
        if solutionViewer:
            self.mbs.SolutionViewer()

        #get sensor data and apply piecewise scaling:
        output = 0*self.GetOutputScaling()
        nDiff = self.nStepsTotal - self.nOutputSteps
        
        output[:,0] = self.mbs.GetSensorStoredData(self.sCoordinates)[1+nDiff:,1] #sensordata includes time
        if self.useVelocities:
            output[:,1] = self.mbs.GetSensorStoredData(self.sCoordinates_t)[1+nDiff:,1] #sensordata includes time
        
        output = self.GetOutputScaling()*output
        return output


class SliderCrank(SimulationModel):
    #initialize class 
    def __init__(self, nStepsTotal=100, useInitialVelocities=False, useInitialAngles=False, 
                 addTime=False, endTime=1, variationLengths=False, useTorqueInput = False,
                 initVelRange = [0,0], initAngleRange = [0,0], flagFlexible=False, 
                 flagResistanceSlider = False, useVelocityInput = False, usePosInput = True, 
                 nCutInputSteps = 0, nOutputSteps = None, tStartup = 0.5, 
                 vMax = 1, aMax = 2, trajType = 0, flagVelNoise = False, outputType = 0):
        
        SimulationModel.__init__(self)

        #required in base:
        self.nStepsTotal = nStepsTotal
        self.nOutputSteps = nOutputSteps
        self.nCutInputSteps = nCutInputSteps
        self.endTime = endTime
        self.tStartup = tStartup 
        if nOutputSteps is None: 
            nOutputSteps = nStepsTotal
            
        self.nOutputSteps = nOutputSteps
        # outputType 0: slider position
        # outputType 1: connecting rod deflection in middle [--> only flexible slider crank!]
        self.outputType = outputType 
        if not(outputType in [0,1]):
            raise ValueError('outputType {} is not implemented/defined!'.format(outputType))
        self.nCutInputSteps = int(np.ceil(tStartup * nStepsTotal / endTime))
        # self.nCutInputSteps= 0 
        
        self.useInitialVelocities = useInitialVelocities
        self.useInitialAngles = useInitialAngles
        self.flagFlexible = flagFlexible
        self.flagResistanceSlider = flagResistanceSlider
        self.variationLengths = variationLengths
        
        
        

        self.smallFact = 1 #0.1    #use small values for inputs to remove tanh nonlinearity
        self.addTime = addTime #add time as input

        self.nODE2 = 2 
        self.m0 = 2 #kg
        self.m1 = 1 #kg
        self.m2 = 1 #kg
        
        self.initVelRange = initVelRange
        self.initAngleRange = initAngleRange
        self.useTorqueInput = useTorqueInput 
        self.useVelocityInput = useVelocityInput
        self.usePosInput = usePosInput
        
        self.factorL1 = 1 #m
        self.factorL2 = 1 #m
        self.gravity = 9.81 #m/s**2
        self.inputUnityVector = True
        
        self.variables = {}
        
        # properties of the velocity trajectories
        if (flagVelNoise or bool(trajType)) and not(useVelocityInput or usePosInput): 
            print('Warning: Velocity trajectory properties set but not velocity input not activated!')
        self.vMax = vMax
        self.aMax = aMax
        self.nPeriods = [20,60]
        self.flagVelNoise = flagVelNoise
        self.trajType = trajType
                                                      
                                                      
        # self.initAngles = np.array([0.5*pi,0.5*pi]) #initial angle from vertical equilibrium
        # self.initAngles = np.array([1.6,2.2]) #initial angle ranges from vertical equilibrium
        # self.initAngles_t = 0*np.array([0.1,0.5]) #initial angular velocity ranges

        #default values, if not varied:
        # self.phi0 = 1.6 #self.smallFact*0.5*pi #will be set later
        # self.phi1 = 2.2 #self.smallFact*0.5*pi #will be set later; WRONG in paper!
        # self.phi0_t = 0 #will be set later
        # self.phi1_t = 0 #will be set later
        
        self.computationType = ModelComputationType.dynamicImplicit

        self.modelName = 'SliderCrank'
        self.modelNameShort = 'slCrank'
        
        self.initHidden = True and self.IsRNN() #for RNN

        #for FFN, we can test also with no initial angles; for RNN it is always needed
        #nInit: 0..1=angle0/1, 2..3=vel0/1
        self.nInit = 1 #  +  self.nODE2*(int(useInitialVelocities) + int(useInitialAngles))
        
        if self.initHidden: 
            self.nInit=0
        
        # self.scalVelocities = 5.
        #inputscaling is difficult for RNN (hidden states), better not to use
        # if self.IsFFN():
        #     self.inputScaling = np.ones(self.nInit + self.variationLengths*2) 
        # else:
        #     self.inputScaling = np.ones((self.nStepsTotal, 2*int(self.variationLengths) + 
        #                                  int(self.addTime))) #2 lengths + time

        self.iDataStarts = int(useInitialVelocities) + int(useInitialAngles)
        self.inputScaling = np.ones(((self.nStepsTotal-self.nCutInputSteps) * (int(useTorqueInput) + int(useVelocityInput) + \
                            int(usePosInput)) + int(useInitialVelocities) + int(useInitialAngles), self.nInit + self.variationLengths*2))
        self.outputScaling = np.ones((self.nOutputSteps, 1)) # 
        
        if usePosInput and self.inputUnityVector: 
            self.fullInputScaling = np.ones([2, self.nStepsTotal])
            self.inputScaling  = self.fullInputScaling[:,self.nCutInputSteps:]
            
        if self.useVelocityInput: 
            self.inputScaling *= self.vMax
            self.outputScaling *= 1.4
        
        if self.outputType == 1: 
            self.outputScaling *= 20
        
        

    # calculate redundant states from minimal coordinates of the crank shaft
    # l1 is the length of the crankshaft, l2 length of the connecting rod
    # the angle phi0 is measured between the crankshaft and the horizontal line 
    # to the slider and phi0_t the corresponding angular velocity
    def CalculateRedundantStates(self, l1, l2, phi0, phi0_t): 
        def GetSliderPosition(phi, l1, l2): 
            return l1*cos(phi) + np.sqrt(l2**2 - sin(phi)**2 * l1**2)
        
        R0 = np.array([[np.cos(phi0), -np.sin(phi0), 0],[np.sin(phi0), np.cos(phi0), 0], [0,0,1]])
        
        x1 = np.sqrt(l2**2 - sin(phi0)**2 * l1**2)
        x2 = GetSliderPosition(phi0, l1, l2) 
        # xMin = GetSliderPosition(np.pi, l1, l2) 
        # xMax = GetSliderPosition(0, l1, l2) 
        phi1 = -np.arctan2(l1*sin(phi0), x1)
        
        # pi = [x_i, y_i, phi_i]
        p0 = [l1/2*cos(phi0), l1/2 *sin(phi0),phi0]
        p1 = [x2 -(l2/2 *cos(phi1)), l1 * sin(phi0) / 2,  phi1]
        p2 = [x2,0]
        
        # velocities: 
        # velocity of the mass (slider)
        x_t   =  (- l1 * np.sin(phi0) - (l1**2*sin(phi0)*cos(phi0))/np.sqrt(l2**2 - l1**2 * np.sin(phi0)**2))*phi0_t
        # translational velocity of the crank shaft is w @ r, where the rotation 
        # matrix is used to calculate the position vector to the 
        v0 = np.cross(np.array([0,0,phi0_t]).T, (R0 @ np.array([l1/2,0, 0]).T))
        # translational velocity of the connecting rod's center of mass (middle) 
        v1 = (v0*2 + [x_t, 0, 0])/2 
        # it can be shown that the velocity of the connecting rod is the average
        # between the two connection point velocities. The translational velocity of 
        # the connecting point with the crank shaft is twice the velcotiy of the 
        # crank shaft's center of mass.  
    
        r1 = RotationMatrixZ(phi1) @ np.array([l2/2,0,0])
        # r1_ = RotationMatrixZ(phi1) @ np.array([-a1,0,0]) # not needed for calculation
        omega_2 = np.cross(r1, [x_t, 0, 0] - v1)
        
        # vi = [v_{xi}, v_{yi}, omega_i]
        v0 = [v0[0],v0[1],phi0_t]
        v1 = [v1[0], v1[1], omega_2[-1]]
        v2 = [x_t, 0]
        return p0, p1, p2, v0, v1, v2
    
    #++++++++++++++++++++++++++++++++
    # slider-crank
    # nonlinear model; index2 and index3-formulation for ConnectorCoordinate and RevoluteJoint2D
    # crank is mounted at (0,0,0); crank length = 2*a0, connecting rod length = 2*a1
    def CreateModel(self):
        self.SC = exu.SystemContainer()
        self.mbs = self.SC.AddSystem()
        self.mbs.variables = self.variables
        
        
        self.timeVecIn = np.arange(0,self.nStepsTotal)/self.nStepsTotal*self.endTime
        self.timeVecOut = np.arange(1,self.nStepsTotal+1)/self.nStepsTotal*self.endTime
        self.mbs.variables['timeVecOut'] = self.timeVecOut

        background = GraphicsDataRectangle(-1, -2, 3, 2, color=[0.9,0.9,0.9,1.])

        oGround = self.mbs.AddObject(ObjectGround(referencePosition= [0,0,0], visualization=VObjectGround(graphicsData= [background])))
        nGround = self.mbs.AddNode(NodePointGround(referenceCoordinates=[0,0,0])) #ground node for coordinate constraint

        #++++++++++++++++++++++++++++++++
        #nodes and bodies

        a0 = 0.25 * self.factorL1     #half x-dim of body
        b0 = 0.05    #half y-dim of body
        massRigid0 = 2
        inertiaRigid0 = massRigid0/12*(2*a0)**2
        graphics0 = GraphicsDataRectangle(-a0,-b0,a0,b0)

        a1 = 0.5 * self.factorL2    #half x-dim of body
        b1 = 0.05    #half y-dim of body
        massRigid1 = 4
        inertiaRigid1 = massRigid1/12*(2*a1)**2
        graphics1 = GraphicsDataRectangle(-a1,-b1,a1,b1)

        phi0, phi0_t = 0,0
        if 'phi0' in self.mbs.variables.keys(): # self.useInitialAngles: 
            phi0 = self.mbs.variables['phi0']
        
        if 'phi0_t' in self.mbs.variables.keys():  # self.useInitialVelocities: 
            phi0_t = self.mbs.variables['phi0_t']
        # phi0, phi0_t = np.pi/2 , 0 # for testing initial conditions can be set here
        # starting conditions
        l1 = a0*2
        l2 = a1*2
        self.l2 = l2
            
        p0, p1, p2, v0, v1, v2 = self.CalculateRedundantStates(l1, l2, phi0, phi0_t)

        nRigid0 = self.mbs.AddNode(Rigid2D(referenceCoordinates=p0, 
                                      initialVelocities=v0));
                                      
        oRigid0 = self.mbs.AddObject(RigidBody2D(physicsMass=massRigid0, 
                                            physicsInertia=inertiaRigid0,
                                            nodeNumber=nRigid0,
                                            visualization=VObjectRigidBody2D(graphicsData= [graphics0])))
        self.mbs.variables['nRigid0'] = nRigid0
        self.mbs.variables['oRigid0'] = oRigid0
        if self.flagFlexible: 
            # parameters of flexible crank
            # note: performed convergence analysis; with 20 elements solution is converged; 
            # for faster calculation also less would be possible. 
            numElements = 20 # 
            E=2.07e8 * 5 #/ 10# * 0.4e-1             # Young's modulus of ANCF element in N/m^2           
            b=0.02                 # width of rectangular ANCF element in m
            h=0.02                  # height of rectangular ANCF element in m
            A=b*h                  # cross sectional area of ANCF element in m^2
            rho = massRigid1/(A*l1)/2 # density of ANCF element in kg/m^3; calculated to fit rigid body model
            I=b*h**3/12            # second moment of area of ANCF element in m^4
            dEI = 1.5e-2*E*I
            dEA = 1e-2*E*A
            self.damping = [dEI, dEA]
            # f=3*E*I/L**2            # tip load applied to ANCF element in N
            # g=-9.81
            preStretch = 0
            cableTemplate = Cable2D(#physicsLength = L / nElements, #set in GenerateStraightLineANCFCable2D(...)
                            physicsLength = l1, 
                            physicsMassPerLength = rho*A,
                            physicsBendingStiffness = E*I,
                            physicsAxialStiffness = E*A,
                            physicsBendingDamping = dEI,
                            physicsAxialDamping = dEA,
                            physicsReferenceAxialStrain = preStretch, #prestretch
                            #nodeNumbers = [0, 0], #will be filled in GenerateStraightLineANCFCable2D(...)
                            visualization=VCable2D(drawHeight=2*h),
                            )
            Rot_1 = RotationMatrixZ(p1[2])[0:2,0:2]
            p1_S = (p1[0:2] + Rot_1 @ [-a1, 0]).tolist() + [0]
            p1_E = (p1[0:2] + Rot_1 @ [ a1, 0]).tolist() + [0]
            ancf = GenerateStraightLineANCFCable2D(self.mbs, p1_S, p1_E, numElements, cableTemplate)
            sensorANCFMiddle = self.mbs.AddSensor(SensorNode(nodeNumber=ancf[0][numElements//2], 
                                                             outputVariableType = exu.OutputVariableType.Position))
            
            # if np.linalg.norm(v1) > 1e-14: 
                # initialization of velocity for ANCF elements is not implemented as written in paper 
                # print('warning: velocity of ANCF element not initialized!') 
        else: 
            nRigid1 = self.mbs.AddNode(Rigid2D(referenceCoordinates = p1, 
                                      initialVelocities    = v1));

 
            oRigid1 = self.mbs.AddObject(RigidBody2D(physicsMass=massRigid1, 
                               physicsInertia=inertiaRigid1,nodeNumber=nRigid1,visualization = VObjectRigidBody2D(
                                                                               graphicsData = [graphics1])))

        c=0.05 #dimension of mass
        sliderMass = 1
        graphics2 = GraphicsDataRectangle(-c,-c,c,c)

        nMass = self.mbs.AddNode(Point2D(referenceCoordinates=p2, initialVelocities=v2))
        oMass = self.mbs.AddObject(MassPoint2D(physicsMass=sliderMass, nodeNumber=nMass,visualization=VObjectRigidBody2D(
                                                                                                graphicsData= [graphics2])))
        
        
        
            
        #++++++++++++++++++++++++++++++++
        # markers for joints:
        # support point # MUST be a rigidBodyMarker, because a torque is applied
        mR0Left = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid0, localPosition=[-a0,0.,0.])) 
        # end point; connection to connecting rod
        mR0Right = self.mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid0, localPosition=[ a0,0.,0.])) 
        
        # markers on first, last and center ANCF element
        if self.flagFlexible:
            mR1LeftC = [0,0]
            mR1RightC = [0,0]
            mR0RightC = [0,0]
            mR2C = [0,0]
            nPoint = self.mbs.AddNode(Point2D(referenceCoordinates=p1_S[0:2]))
            mPoint = self.mbs.AddMarker(MarkerNodePosition(nodeNumber=nPoint))
            self.mbs.AddObject(RevoluteJoint2D(markerNumbers=[mR0Right, mPoint]))
            for i in range(2): 
                mR0RightC[i] = self.mbs.AddMarker(MarkerNodeCoordinate(name='mR0Right' + str(i), nodeNumber=nPoint, 
                                                                       coordinate=i))
                mR1LeftC[i] =  self.mbs.AddMarker(MarkerNodeCoordinate(name='mR1LeftC' + str(i), nodeNumber = ancf[0][0], 
                                                                       coordinate=i)) 
                mR1RightC[i] = self.mbs.AddMarker(MarkerNodeCoordinate(name='mR0RightC' + str(i), nodeNumber = ancf[0][-1],
                                                                       coordinate=i)) 
                mR2C[i] =      self.mbs.AddMarker(MarkerNodeCoordinate(name='mR2C' + str(i), nodeNumber = nMass, coordinate=i)) 
            


        else: 
            mR1Left = self.mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid1, localPosition=[-a1,0.,0.])) #connection to crank
            mR1Right = self.mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid1, localPosition=[ a1,0.,0.])) #end point; connection to slider

        mMass = self.mbs.AddMarker(MarkerBodyPosition(bodyNumber=oMass, localPosition=[ 0.,0.,0.]))
        mG0 = self.mbs.AddMarker(MarkerBodyPosition(bodyNumber=oGround, localPosition=[0,0,0.]))

        #++++++++++++++++++++++++++++++++
        #joints:
        self.mbs.AddObject(RevoluteJoint2D(name ='jointGroundCrankshaft', markerNumbers=[mG0,mR0Left]))
        
        # flexible joint uses 2 coordinate constraints which is equivalent to a single revolutJoint2D
        if self.flagFlexible: 
            for i in range(2): 
                self.mbs.AddObject(CoordinateConstraint(markerNumbers=[mR0RightC[i],mR1LeftC[i]])) # add constraint  
                self.mbs.AddObject(CoordinateConstraint(markerNumbers=[mR1RightC[i],mR2C[i]])) 
        else: 
            self.mbs.AddObject(RevoluteJoint2D(markerNumbers=[mR0Right,mR1Left]))
            self.mbs.AddObject(RevoluteJoint2D(markerNumbers=[mR1Right,mMass]))

        #++++++++++++++++++++++++++++++++
        #markers for node constraints:
        mGround = self.mbs.AddMarker(MarkerNodeCoordinate(name='mGroundCoordinate', nodeNumber = nGround, coordinate=0)) #Ground node ==> no action
        mNodeSlider = self.mbs.AddMarker(MarkerNodeCoordinate(nodeNumber = nMass, coordinate=1)) #y-coordinate is constrained
        
        #++++++++++++++++++++++++++++++++
        #coordinate constraints
        self.mbs.AddObject(CoordinateConstraint(markerNumbers=[mGround,mNodeSlider]))
    
            
        #loads and driving forces:
        def UFLoad(mbs, t, loadVector): 
            # iVal = (t * mbs.variables['nStepsTotal'])
            # print('t: ', t, 'ival: ', iVal, 'max: ', len(mbs.variables ['inputTorque']))
            # val = mbs.variables['inputTorque'][int(iVal)][0]
            # val  = 10
            val =  10 + 0*50 *GetInterpolatedSignalValue (t, mbs.variables['inputTorque'], 
                                             mbs.variables['timeVecOut'],rangeWarning=False)
            # val = 0
            return [0,0,val]
            
        self.debugData = []
        def PrestepControlVelocity(mbs, t): 
            # get current state of system
            velCurrent = mbs.GetNodeOutput(mbs.variables['nRigid0'], exu.OutputVariableType.Coordinates_t, exu.ConfigurationType.Current)[2]
            phiCurrent = mbs.GetNodeOutput(mbs.variables['nRigid0'], exu.OutputVariableType.Coordinates)[2]
            
            if t == 0: 
                # initialize angle in first step!
                mbs.variables['phiDesired'] = phiCurrent
                return True
            
            if t > 2: 
                pass
            
            dt = mbs.sys['dynamicSolver'].it.currentStepSize
            velDesired = GetInterpolatedSignalValue (t, mbs.variables['inputVel'],
                                       mbs.variables['timeVecOut'],rangeWarning=False)
            mbs.variables['phiDesired'] = mbs.variables['phiDesired']  + velDesired * dt 

            p = 10
            d = 100
            myTorque = p * (mbs.variables['phiDesired'] - phiCurrent) +  d * (velDesired - velCurrent)
            mbs.SetLoadParameter(self.mbs.variables['lTorque'], 'loadVector', [0,0, myTorque])
            self.debugData += [[mbs.variables['phiDesired'], phiCurrent, velDesired, velCurrent]]

            return True
        
        def PrestepControlPosition(mbs, t): 
            # get current state of system
            velCurrent = mbs.GetNodeOutput(mbs.variables['nRigid0'], exu.OutputVariableType.Coordinates_t, exu.ConfigurationType.Current)[2]
            phiCurrent = mbs.GetNodeOutput(mbs.variables['nRigid0'], exu.OutputVariableType.Coordinates)[2] + mbs.variables['phi0']
            # sems to always start at 0??
            
            dt = mbs.sys['dynamicSolver'].it.currentStepSize
            posDesired = GetInterpolatedSignalValue (t, mbs.variables['inputPos'],
                                       mbs.variables['timeVecOut'],rangeWarning=False)
            posDesiredNext = GetInterpolatedSignalValue (t+dt, mbs.variables['inputPos'],
                                       mbs.variables['timeVecOut'],rangeWarning=False)
            velDesired = (posDesiredNext - posDesired) / dt



            p = 10
            d = 100
            myTorque = p * (posDesired - phiCurrent) +  d * (velDesired - velCurrent)
            mbs.SetLoadParameter(self.mbs.variables['lTorquePos'], 'loadVector', [0,0, myTorque])
            self.debugData += [[posDesired, phiCurrent, velDesired, velCurrent]]
            if t <= dt and False: 
                # initialize angle in first step!
                # mbs.variables['phiDesired'] = phiCurrent
                print('phiCurrent: ', phiCurrent)
                print('mbs.variables phi0: ', mbs.variables['phi0'])
                print('phiDesired: ', posDesired)
            return True
        
        if self.flagResistanceSlider: 
            def UFSliderForce(t, mbs, load): 
                v = self.mbs.GetNodeOutput(nMass, exu.OutputVariableType.Velocity, exu.ConfigurationType.Current)
                F = -3*StribeckFunction(v[0], muDynamic=1, muStaticOffset=1.5, regVel=1e-3)
                # F = 10/ 0.5 #0.8660254037844386 
                # print(F)
                return F
            mSlider0 = self.mbs.AddMarker(MarkerNodeCoordinate(nodeNumber= nMass, coordinate = 0)) 
            self.mbs.AddLoad(LoadCoordinate(markerNumber=mSlider0, load=0, loadUserFunction=UFSliderForce)) 
            
        if self. useTorqueInput: 
            self.mbs.AddLoad(Torque(markerNumber = mR0Left, loadVector = [0, 0, 10], loadVectorUserFunction = UFLoad)) #apply torque at crank
        
        elif self.useVelocityInput: 
            self.mbs.variables['lTorque'] = self.mbs.AddLoad(Torque(markerNumber = mR0Left, loadVector = [0, 0, 0]))
            self.mbs.variables['phiDesired'] = phi0
            
            self.mbs.SetPreStepUserFunction(PrestepControlVelocity)
            
        elif self.usePosInput: 
            if 'inputPos_e2' in self.mbs.variables:
                # calculate angles from unity vectors
                phiInput = np.arctan2(self.mbs.variables['inputPos_e2'], self.mbs.variables['inputPos_e1'])
                self.mbs.variables['inputPos'] =  AccumulateAngle(phiInput)
                
            self.mbs.variables['lTorquePos'] = self.mbs.AddLoad(Torque(markerNumber = mR0Left, loadVector = [0, 0, 0]))
            self.mbs.SetPreStepUserFunction(PrestepControlPosition)
            
        

        # #++++++++++++++++++++++++++++++++
        #assemble, adjust settings and start time integration
        self.mbs.Assemble()
        # exu.StartRenderer()
        # self.mbs.WaitForUserToContinue()
        if self.useVelocityInput: 
            PrestepControlVelocity(self.mbs, 0) # initialize desired angle for first step with current angle
            

        #now as system is assembled, nodes know their global coordinate index (for reading the coordinate out of the solution file):
        self.nMass = self.mbs.GetNodeODE2Index(nMass) 
        #alternatively: use mbs.systemData.GetObjectLTGODE2(oMass)[0] to obtain e.g. first coordinate index of sliding mass object

        self.simulationSettings = exu.SimulationSettings() # takes currently set values or default values
        
        # simulation steps is factor of 10 higher than the output for training the NN
        self.simulationSettings.timeIntegration.numberOfSteps = self.GetNSimulationSteps()
        
        
        self.simulationSettings.timeIntegration.endTime = self.endTime
        self.simulationSettings.timeIntegration.verboseMode = 1 

        self.simulationSettings.solutionSettings.sensorsWritePeriod = self.endTime/self.nStepsTotal
        
        self.simulationSettings.timeIntegration.newton.useModifiedNewton = True
        
        # simulationSettings.timeIntegration.generalizedAlpha.useNewmark = True
        # simulationSettings.timeIntegration.generalizedAlpha.useIndex2Constraints = True
        self.simulationSettings.timeIntegration.generalizedAlpha.spectralRadius = 0.5
        self.simulationSettings.timeIntegration.newton.absoluteTolerance = 1e-6
        # self.simulationSettings.timeIntegration.newton.relativeTolerance = 1e-{}
        
        self.sPos2 = self.mbs.AddSensor(SensorBody(bodyNumber=oMass, storeInternal=True,
                                              outputVariableType=exu.OutputVariableType.Position))
        sBody0 = self.mbs.AddSensor(SensorBody(bodyNumber=oRigid0, localPosition = [a0,0,0], 
                                                          outputVariableType = exu.OutputVariableType.Position))
        sBody0Rot = self.mbs.AddSensor(SensorBody(bodyNumber=oRigid0, localPosition = [a0,0,0], 
                                                          outputVariableType = exu.OutputVariableType.Rotation))
        self.mbs.variables['sBody0'] = sBody0 
        self.mbs.variables['sBody0Rot'] = sBody0Rot
        
        if self.outputType == 1: 
            # only works with flexible bodies! 
            if not(self.flagFlexible): 
                raise print("Warning: outputtype {} not appropriate for rigid bodies!".format(self.outputType))
            
            
            def UFSensorOutputType1(mbs, t, sensorNumbers, factors, configuration): 
                p0 = mbs.GetSensorValues(sensorNumbers[0])
                p2 = mbs.GetSensorValues(sensorNumbers[1])
                pANCF = mbs.GetSensorValues(sensorNumbers[2])
                p02 = p0 - p2
                phiObject = np.arctan2(p02[0], p02[1])
                pMiddle = (p0 + p2)/2 # desired position in the middle
                pDiff = (pANCF - pMiddle)[0:2]
                e_ = np.array([np.cos(phiObject+np.pi/2), np.sin(phiObject+np.pi/2)])
                deflection = pDiff @ e_ # project in direction  of normal to ideal angle
                return [deflection]
            
            
            self.sDeflection = self.mbs.AddSensor(SensorUserFunction(sensorNumbers=[sBody0, self.sPos2, sensorANCFMiddle],
                                 # fileName='solution/sensorTest2.txt',
                                 storeInternal=True,  sensorUserFunction=UFSensorOutputType1))

        # Sensors
        # self.sPos0 = self.mbs.AddSensor(SensorBody(bodyNumber=o0, storeInternal=True,
        #                                       outputVariableType=exu.OutputVariableType.Position))
        # self.sPos1 = self.mbs.AddSensor(SensorBody(bodyNumber=o1, storeInternal=True,
        #                                       outputVariableType=exu.OutputVariableType.Position))
        # self.sVel0 = self.mbs.AddSensor(SensorBody(bodyNumber=o0, storeInternal=True,
        #                                       outputVariableType=exu.OutputVariableType.Velocity))
        # self.sVel1 = self.mbs.AddSensor(SensorBody(bodyNumber=o1, storeInternal=True,
        #                                       outputVariableType=exu.OutputVariableType.Velocity))
        
    


    #get time vector according to output data
    def GetOutputXAxisVector(self):
        if self.nOutputSteps != self.nStepsTotal: 
            return self.timeVecOut[self.nStepsTotal-self.nOutputSteps:]    
        else: 
            return self.timeVecOut

    def getInputScaling(self):     
        return self.inputScalingFactor*self.inputScaling
    
    #create a randomized input vector
    #relCnt can be used to create different kinds of input vectors (sinoid, noise, ...)
    #isTest is True in case of test data creation
    def CreateInputVector(self, relCnt = 0, isTest=False, dataErrorEstimator = False):
        # if dataErrorEstimator: 
        #     raise ValueError('Todo: implement dataErrorEstimator!')
        if self.useVelocityInput and self.useTorqueInput: 
            raise ValueError('Do NOT use torque and velocity input simultaniously!')
            
        inputData = {}
        vec = np.zeros(self.GetInputScaling().shape)
        #print(vec.shape)
        lengths = []
        if self.variationLengths:
            lengths = np.zeros(2)

            lengths[0] = 1.+1.*np.random.rand() #L0
            lengths[1] = 2.+1.*np.random.rand() #L1
        t = np.linspace(0, self.endTime, self.nStepsTotal)
        # w = 0.5 + 0.5*np.random.random()
        # todo: change w and/or add a phi_0
        w = (4*np.pi)
        if self.useInitialAngles: 
            inputData['phi0'] = np.random.uniform(self.initAngleRange[0], self.initAngleRange[1])
    
        if self.useInitialVelocities: 
            inputData['phi0_t']= np.random.uniform(self.initVelRange[0], self.initVelRange[1])
            
        if self.useTorqueInput: 
            inputData['inputTorque'] = np.random.rand() * np.ones(self.nStepsTotal) * np.sin(w*t.T) 

        if self.useVelocityInput: 
            inputData['inputVel'] = CreateVelocityProfile(self.tStartup, self.endTime, self.nStepsTotal, vMax = self.vMax*(1 + bool(dataErrorEstimator)*0.5), 
                                                          aMax = self.aMax *(1 + bool(dataErrorEstimator)*0.5), 
                                                          nPeriods = self.nPeriods, flagNoise = self.flagVelNoise, trajType = self.trajType)
            
        if self.usePosInput: 
            velProfile = CreateVelocityProfile(self.tStartup, self.endTime, self.nStepsTotal, vMax = self.vMax *(1+ bool(dataErrorEstimator)*0.5), 
                                               aMax = self.aMax *(1+ bool(dataErrorEstimator)*0.5), 
                                               nPeriods = self.nPeriods, flagNoise = self.flagVelNoise, trajType = self.trajType)
            
            phi =  numericIntegrate(velProfile, self.endTime/self.nStepsTotal)
            if self.useInitialAngles: 
                phi = np.array(phi) + inputData['phi0']
            if self.inputUnityVector: # 
                inputData['inputPos_e1'] = np.cos(phi)
                inputData['inputPos_e2'] = np.sin(phi)
            else: # for neural network unity vector is probably better... 
                # inputData['inputPos'] = 
                inputData['inputPos'] = phi
            # np.random.rand() * np.ones(self.nStepsTotal) * np.sin(w*t.T) 

        # self.torqueFcat*(np.random.rand(2))
        # print(randTorque)
        # randVels = self.smallFact*self.initAngles_t*np.random.rand(2)

        #for RNN, we can also avoid variations:
        # if not self.useInitialAngles:
        #     randAngles = self.initAngles
        # if not self.useInitialVelocities:
        #     randVels = [0.,0.]

        # if not self.IsFFN():
        #     for i, length in enumerate(lengths):
        #         vec[:,i] = length
        #     if self.addTime:
        #         vec[:,0+len(lengths)] = self.timeVecOut
        # else:
        #     off = 0
        #     if self.useInitialAngles:
        #         vec[0] = randAngles[0]
        #         vec[1] = randAngles[1]
        #         off += 2
        #     if self.useInitialVelocities:
        #         vec[0+off] = randVels[0]
        #         vec[1+off] = randVels[1]
        #         off += 2
    
        #     for i, length in enumerate(lengths):
        #         vec[off+i] = length
        #     off += len(lengths)

        # vec[:,0] = randTorque
        vec = self.MergeInputData(inputData)
        
        return vec

    #create initialization of (couple of first) hidden states (RNN)
    def CreateHiddenInit(self, isTest):
        if self.initHidden:
            vec = np.zeros(2*self.nODE2)
            randAngles = self.smallFact*self.initAngles * (2*np.random.rand(self.nODE2)-1.) 
            randVels = self.smallFact*self.initAngles_t*np.random.rand(self.nODE2)
            
            vec[0:2] = randAngles
            vec[2:4] = randVels

            return vec
        else:
            return np.array([])
            
    #get number of simulation steps
    def GetNSimulationSteps(self):
        if self.flagFlexible: 
            return self.nStepsTotal*20  
        else: 
            return self.nStepsTotal*10 #10 x finer simulation than output

    # merge dictionary into vector
    def MergeInputData(self, inputData):
        vec = np.array([[]])
        if self.useInitialAngles and not(self.inputUnityVector): 
            vec = np.concatenate((vec, [[inputData['phi0']]]), axis=1)
        if self.useInitialVelocities: 
            vec = np.concatenate((vec, [[inputData['phi0_t']]]), axis=1)
        if self.useTorqueInput: 
            vec = np.concatenate((vec, [inputData['inputTorque']]), axis=1)
        if self.useVelocityInput:
            vec = np.concatenate((vec, [inputData['inputVel']]), axis=1)
        if self.usePosInput: 
            if self.inputUnityVector: 
                vec = np.concatenate(([inputData['inputPos_e1']], [inputData['inputPos_e2']]), axis=0)    
            else: 
                vec = np.concatenate((vec, [inputData['inputPos']]), axis=1)
            
            
        return vec
    
    # split input data into initial values, forces or other inputs
    # return dict with 'data' and possibly 'initialODE2' and 'initialODE2_t'
    def SplitInputData(self, inputData, hiddenData=None):
        # data = np.array(self.GetInputScaling()*inputData)
        inputDict = {}
        iIndex = 0
        if self.useInitialAngles and not(self.inputUnityVector): 
            inputDict['phi0'] = inputData[0][iIndex]
            iIndex += 1
            
        if self.useInitialVelocities: 
            inputDict['phi0_t'] = inputData[0][iIndex]
            iIndex += 1
        if self.useTorqueInput: 
            inputDict['inputTorque'] = inputData[0][iIndex:iIndex+self.nStepsTotal]
            iIndex += self.nStepsTotal
        
        if self.useVelocityInput: 
            inputDict['inputVel'] = inputData[0][iIndex:iIndex+self.nStepsTotal]
            iIndex += self.nStepsTotal
        
        if self.usePosInput:
            if self.inputUnityVector: 
                inputDict['inputPos_e1']  = inputData[0,:]
                inputDict['inputPos_e2']  = inputData[1,:]
                inputDict['phi0'] = np.arctan2(inputDict['inputPos_e2'][0], inputDict['inputPos_e1'][0])
            else: 
                inputDict['inputPos'] = inputData[0][iIndex:iIndex+self.nStepsTotal]
                iIndex += self.nStepsTotal
                
            # if self.addTime:
            #     rv['time'] = data[off:] #lengths are the same for all sequences
        
        return inputDict # rv

    #split output data to get ODE2 values (and possibly other data, such as ODE2)
    #return dict {'ODE2':[], 'ODE2_t':[]}
    def SplitOutputData(self, outputData):
        dataDict = {}
        dataDict['x'] = outputData
        

        return dataDict
     
    #convert all output vectors into plottable data (e.g. [time, x, y])
    #the size of data allows to decide how many columns exist
    def OutputData2PlotData(self, outputData, forSolutionViewer=False):
        timeVec = self.GetOutputXAxisVector()
        dataDict = self.SplitOutputData(outputData)
        
        data = np.vstack((timeVec, dataDict['x'].T)), #dataDict['phi1'].T,
                          # dataDict['phi0_t'].T, dataDict['phi1_t'].T)).T
            
        return data

    #return dict of names to columns for plotdata        
    def PlotDataColumns(self):
        d = {'time':0, 'x':1} 
        # time is appended automatically; after that the model output is shown
        
        return d

    #get compute model with given input data and return output data
    #initialState contains position and velocity states as list of two np.arrays 
    def ComputeModel(self, inputData, hiddenData=None, verboseMode = 0, solutionViewer = False):
        #set input data ...
        
        # inputDict = self.SplitInputData(np.array(inputData), hiddenData)
        # inputDict = self.SplitInputData(inputData)
        inputDict = self.SplitInputData(self.fullInputScaling * np.array(inputData))
        # if len(inputData) == 2: 
        for key, value in inputDict.items(): 
            self.variables[key] = value
        
        # if self.inputUnityVector: 
            # self.variables[]
            # pass
            
        self.CreateModel() #must be created newly for each test ...
        self.mbs.variables['nStepsTotal'] = self.nStepsTotal
        # self.mbs.variables['inputTorque'] = inputData
        
        
        self.simulationSettings.timeIntegration.verboseMode = verboseMode
        self.simulationSettings.solutionSettings.writeSolutionToFile = solutionViewer 
        
        self.mbs.Assemble()
        # exu.StartRenderer()
        # self.mbs.WaitForUserToContinue()
        self.mbs.SolveDynamic(self.simulationSettings, showHints=True) #GeneralizedAlpha

        if solutionViewer:
            self.mbs.SolutionViewer()

        #get sensor data and apply piecewise scaling:
        # output = 0*self.GetOutputScaling()
        
        if self.outputType == 0: 
            output = self.mbs.GetSensorStoredData(self.sPos2)[1:,1] # x position of slider
            if self.nOutputSteps != self.nStepsTotal: 
                output = output[self.nStepsTotal-self.nOutputSteps:]
            
            output = self.GetOutputScaling()*np.array([output]).T - self.l2
        elif self.outputType == 1: 
            output = self.mbs.GetSensorStoredData(self.sDeflection)[1:,1]
            output = self.GetOutputScaling() * np.array([output[self.nStepsTotal-self.nOutputSteps:]]).T
        else: 
            raise ValueError('outputType {} is not implemented/defined!'.format(self.outputType))
        return output

    def SolutionViewer(self, outputData, outputDataColumns = [0]):
        #model is 2D!
        #NOTE: L0 and L1 may be wrong! this is just for visualization!
        nColumns = 2*2 #2 x (x,y)
        angles = self.OutputData2PlotData(outputData, forSolutionViewer=True)
        #print(angles)
        data = np.zeros((self.nStepsTotal, 1+nColumns))
        for i, t in enumerate(self.timeVecOut):
            data[i,0] = t
            data[i,1] = self.L0*sin(angles[i,1])
            data[i,2] = +self.L0*(1-cos(angles[i,1]))
            data[i,3] = data[i,1]+self.L1*sin(angles[i,2])
            data[i,4] = data[i,2]-self.L1*cos(angles[i,2])+self.L1
        
        # print(data)
        
        # columnsExported = dict({'nODE2':self.nODE2, 
        #                         'nVel2':0, 'nAcc2':0, 'nODE1':0, 'nVel1':0, 'nAlgebraic':0, 'nData':0})
        columnsExported = [nColumns, 0, 0, 0, 0, 0, 0] #nODE2 without time
        if data.shape[1]-1 != nColumns:
            raise ValueError('SimulationModel.SolutionViewer: problem with shape of data: '+
                             str(nColumns)+','+str(data.shape))

        nRows = data.shape[0]
        
        
        sol = dict({'data': data, 'columnsExported': columnsExported,'nColumns': nColumns,'nRows': nRows})

        self.mbs.SolutionViewer(sol,runOnStart=True)

        
#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# model of the 6R Robot "Puma", standing on a flexible socket
class Flex6RRobot(SimulationModel):
    #initialize class 
    def __init__(self, nStepsTotal=100, endTime=1, 
                 stepTime=1, nModes=8, isRigid=False, flagFlexibleArm = True, 
                 createModel=True, verboseMode = 0, inputType = 0, outputType = 0, 
                 nOutputSteps = None,  EModulus = 0.1e9):
        SimulationModel.__init__(self)

        self.nStepsTotal = int(nStepsTotal)
        if nOutputSteps is None: 
            self.nOutputSteps =int(nStepsTotal)
        else: 
            self.nOutputSteps = int(nOutputSteps)
        
        
        self.createMesh = createModel
        self.nModes = nModes
        self.endTime = endTime
        self.stepTime = stepTime # time at which angle is changed from 0 to target
        self.isRigid = isRigid
        self.useFlexBody = not(isRigid)
        
        self.velocityGround = False
        self.flagFlexibleArm = flagFlexibleArm
        self.inputType = inputType 
        self.outputType = outputType
        
        self.computedTorqueControl = False
        self.computeStatic = True
        
        if self.computedTorqueControl: 
            import roboticstoolbox as rtb
        
        self.nJoints = 6
        self.qLim = [[-2.7925268 , -1.91986218, -2.35619449, -pi*0.99, -1.74532925, -pi*0.99], 
                     [ 2.7925268 ,  1.91986218,  2.35619449,  pi*0.99,  1.74532925,  pi*0.99]]
        # self.nOutputPos = 1 #number of 3D output acc / positions which are prediced

        self.verboseMode = verboseMode
        self.computationType = ModelComputationType.dynamicImplicit

        self.modelName = 'FFRF6RArm'
        self.modelNameShort = 'nl-ffrf6R'
        # according to: Sezimara et al. 2002, An optimum robot path planning with payload constraints
        
        self.vMax = np.array([ 82,  74, 122, 228, 241, 228])*np.pi/180
        self.aMax = np.array([286, 286, 572, 572, 572, 572])*np.pi/180
        self.EModulus  = EModulus 
        # max torque: [77,133,66,13,12,13]
        
        scalInputs = 0.5*pi #+/- angle that is prescribed to torsional spring dampers
        # inputType 0: 
        if inputType == 0: 
            self.inputScaling = np.ones(6) * pi
        elif inputType == 1: 
            self.inputScaling = np.ones([6,2]) * pi
        # from qStart to qEnd with all intermediate angles
        elif inputType in [2,3,4,5]: 
            self.inputScaling = np.ones([self.nStepsTotal, 6]) * pi
            
        else: 
            self.inputScaling = scalInputs*np.ones(self.nJoints*self.nStepsTotal)

        # outputType 0: all positions
        if outputType == 0: 
            self.outputScaling = np.ones([self.nStepsTotal, 3])
            
        # all positions and rotations
        elif outputType == 1: 
            self.outputScaling = np.ones([self.nStepsTotal, 6])

            # self.outputScaling = np.array([20.]*(3*self.nStepsTotal)+ #ground velocities scaled higher!
            #                               [1.]*(3*self.nStepsTotal) )
        # positioning error because of flexibility
        elif outputType == 2: 
            self.outputScaling = np.ones([self.nStepsTotal, 3]) *[10,10,30] # ,1,1,1]
            if self.EModulus > 1e8: 
                self.outputScaling *= self.EModulus/1e8
            
        elif outputType == 3: 
            self.outputScaling = np.ones([self.nStepsTotal, 6]) *[10,10,30,10,10,10]
            
        else: 
            self.outputScaling = np.ones((self.nStepsTotal, 3*self.nOutputPos))*40
        self.createMesh = True


        # cut outputscaling for asymetric windows
        if not(self.nOutputSteps is None) and not (self.nOutputSteps == self.nStepsTotal): 
            self.outputScaling = self.outputScaling[-self.nOutputSteps:, :]
        # if self.velocityGround:
            # self.outputScaling[:,0:3]*= 200. #scale ground velocities
        
        
    def CreateModel(self, q0 = [0,0,0,0,0,0], flagComputeModel = False, computeStatic = False): 
        self.SC = exu.SystemContainer()
        self.mbs = self.SC.AddSystem()
        
        self.timeVecIn = np.arange(0,self.nStepsTotal)/self.nStepsTotal*self.endTime
        self.timeVecOut = np.arange(1,self.nStepsTotal+1)/self.nStepsTotal*self.endTime
        
        sensorWriteToFile = False
        
        #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++
        gravity=[0,0,-9.81]
        #geometry 
        L = [0.075,0.4318,0.15005,0.4318]
        W = [0,0,0.0203,0]
        rArm = 0.025 #radius arm
        rFlange = 0.05 #radius of flange
        meshSize = rArm*2
        meshOrder = 1 # 2 is more accurate but slower... 
        useFlexBody = self.useFlexBody
        Emodulus= self.EModulus 
        dampingK = 8e-3 #stiffness proportional damping
        self.dampingK = dampingK
        fileNames = ['meshes/netgenRobotBase_meshSize{}_meshOrder{}_E{}_d{}'.format(meshSize, meshOrder, Emodulus, dampingK),
                     # here additional files could be added, e.g. for adding a flexible link to the robot.
                     ] #to load/save of FEM data
        try: 
            for file in os.listdir('meshes'): 
                if fileNames[0].split('/')[1] in file: 
                    self.createMesh = False
        except: 
            self.createMesh = True
        Lbase = 0.3
        flangeBaseR = 0.05 #socket of base radius
        flangeBaseL = 0.05 #socket of base length
        rBase = 0.05
        tBase = 0.01 #wall thickness
        
        # values for mesh:
        nModes = self.nModes # 12 # 8
        rho = 1000
        nu=0.3
        
        nFlexBodies = 1*int(useFlexBody)
        femList = [None]*nFlexBodies
        
        def GetCylinder(p0, axis, length, radius):
            pnt0 = Pnt(p0[0], p0[1], p0[2])
            pnt1 = pnt0 + Vec(axis[0]*length, axis[1]*length, axis[2]*length)
            cyl = Cylinder(pnt0, pnt1, radius)
            plane0 = Plane (pnt0, Vec(-axis[0], -axis[1], -axis[2]) )
            plane1 = Plane (pnt1, Vec(axis[0], axis[1], axis[2]) )
            return cyl*plane0*plane1
        
        
        fb=[] #flexible bodies list of dictionaries
        fb+=[{'p0':[0,0,-Lbase], 'p1':[0,0,0], 'axis0':[0,0,1], 'axis1':[0,0,1]}] #defines flanges
        
        fes = None
        
        #create flexible bodies
        #requires netgen / ngsolve
        #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.createMesh and self.useFlexBody: #needs netgen/ngsolve to be installed to compute mesh, see e.g.: https://github.com/NGSolve/ngsolve/releases
            femList[0] = FEMinterface()
            # for older ngsolve versions this is / was needed when no installer was available. 
            # import sys
            # adjust path to your ngsolve installation (if not added to global path)
            # sys.path.append('C:/ProgramData/ngsolve/lib/site-packages') 

            #++++++++++++++++++++++++++++++++++++++++++++++++
            #flange
            geo = CSGeometry()

            geo.Add(GetCylinder(fb[0]['p0'], fb[0]['axis0'], Lbase-flangeBaseL, rBase) - 
                    GetCylinder([0,0,-Lbase+tBase], [0,0,1], Lbase-2*tBase-flangeBaseL, rBase-tBase) + 
                    GetCylinder([0,0,-flangeBaseL-tBase*0.5], fb[0]['axis1'], flangeBaseL+tBase*0.5, flangeBaseR))
        
            print('start meshing flexible socket for 6R')
            mesh = ngs.Mesh( geo.GenerateMesh(maxh=meshSize))
            mesh.Curve(1)
            print('finished meshing')
        
            if False: #set this to true, if you want to visualize the mesh inside netgen/ngsolve; only for debugging. 
                # import netgen
                import netgen.gui
                ngs.Draw(mesh)
                for i in range(10000000):
                    netgen.Redraw() #this makes the window interactive
                    time.sleep(0.05)
        
            #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++
            #Use fem to import FEM model and create FFRFreducedOrder object
            [bfM, bfK, fes] = femList[0].ImportMeshFromNGsolve(mesh, density=rho, youngsModulus=Emodulus, poissonsRatio=nu, meshOrder=meshOrder)
            femList[0].SaveToFile(fileNames[0])
            
        elif not(self.createMesh)  and self.useFlexBody: 
            femList[0] = FEMinterface()
            femList[0].LoadFromFile(fileNames[0])
        

        #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++
        #compute flexible modes
        if self.createMesh and self.useFlexBody: 
            for i in range(nFlexBodies):
                fem = femList[i]
                nodesPlane0 = fem.GetNodesInPlane(fb[i]['p0'], fb[i]['axis0'])
                lenNodesPlane0 = len(nodesPlane0)
                weightsPlane0 = np.array((1./lenNodesPlane0)*np.ones(lenNodesPlane0))
                
                
                nodesPlane1  = fem.GetNodesInPlane(fb[i]['p1'], fb[i]['axis1'])
                centerPointAverage0 = fem.GetNodePositionsMean(nodesPlane0)
                centerPointAverage1 = fem.GetNodePositionsMean(nodesPlane1)
                
                # print('body'+str(i)+'nodes1=', nodesPlane1)
                lenNodesPlane1 = len(nodesPlane1)
                weightsPlane1 = np.array((1./lenNodesPlane1)*np.ones(lenNodesPlane1))
                
                boundaryList = [nodesPlane0, nodesPlane1] 
                
                print("nNodes=",fem.NumberOfNodes())
                
                print("compute flexible modes... ")
                start_time = time.time()
                fem.ComputeHurtyCraigBamptonModes(boundaryNodesList=boundaryList, 
                                                  nEigenModes=nModes, 
                                                  useSparseSolver=True,
                                                  computationMode = HCBstaticModeSelection.RBE2)
                print("compute modes needed %.3f seconds" % (time.time() - start_time))
                
            #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++
            #compute stress modes for postprocessing (inaccurate for coarse meshes, just for visualization):
            # if fes != None:
                mat = KirchhoffMaterial(Emodulus, nu, rho)
                varType = exu.OutputVariableType.StressLocal
                #varType = exu.OutputVariableType.StrainLocal
                print("ComputePostProcessingModes ... (may take a while)")
                start_time = time.time()
                
                #without NGsolve, but only for linear elements
                # fem.ComputePostProcessingModes(material=mat, 
                #                                outputVariableType=varType)
                fem.ComputePostProcessingModesNGsolve(fes, material=mat, 
                                               outputVariableType=varType)
        
                print("   ... needed %.3f seconds" % (time.time() - start_time))
                # SC.visualizationSettings.contour.reduceRange=False
                self.SC.visualizationSettings.contour.outputVariable = varType
                self.SC.visualizationSettings.contour.outputVariableComponent = -1 #x-component
                femList[i].SaveToFile(fileNames[i])
                self.createMesh = False
                cms = ObjectFFRFreducedOrderInterface(fem)
                
                objFFRF = cms.AddObjectFFRFreducedOrder(self.mbs, positionRef=[0,0,0], 
                                                        initialVelocity=[0,0,0], 
                                                        initialAngularVelocity=[0,0,0],
                                                        stiffnessProportionalDamping=dampingK,
                                                        gravity=gravity,
                                                        color=[0.1,0.9,0.1,1.],
                                                        )
                mPlane0 = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=objFFRF['oFFRFreducedOrder'], 
                                                      meshNodeNumbers=np.array(nodesPlane0), #these are the meshNodeNumbers
                                                      weightingFactors=weightsPlane0, 
                                                      offset=-centerPointAverage0 - [0,0,Lbase]))
                                      
                mPlane1 = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=objFFRF['oFFRFreducedOrder'], 
                                                              meshNodeNumbers=np.array(nodesPlane1), #these are the meshNodeNumbers
                                                              weightingFactors=weightsPlane1, 
                                                              offset=-centerPointAverage1))
                
                if i==0:
                    baseMarker = mPlane1
                    oGround = self.mbs.AddObject(ObjectGround(referencePosition= [0,0,0]))
                    mGround = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition=fb[i]['p0']))
                    self.mbs.AddObject(GenericJoint(markerNumbers=[mGround, mPlane0], 
                                               constrainedAxes = [1,1,1,1,1,1],
                                               visualization=VGenericJoint(axesRadius=rFlange*0.5, axesLength=rFlange)))
                    
        elif not(self.createMesh) and self.useFlexBody: 
            for i in range(nFlexBodies):     
                fem = femList[i] 
                fem.LoadFromFile(fileNames[i])
                cms = ObjectFFRFreducedOrderInterface(fem)
                
                objFFRF = cms.AddObjectFFRFreducedOrder(self.mbs, positionRef=[0,0,0], 
                                                        initialVelocity=[0,0,0], 
                                                        initialAngularVelocity=[0,0,0],
                                                        stiffnessProportionalDamping=dampingK,
                                                        gravity=gravity,
                                                        color=[0.1,0.9,0.1,1.],
                                                        )
                
                nodesPlane0 = fem.GetNodesInPlane(fb[i]['p0'], fb[i]['axis0'])
                # print('body'+str(i)+'nodes0=', nodesPlane0)
                lenNodesPlane0 = len(nodesPlane0)
                weightsPlane0 = np.array((1./lenNodesPlane0)*np.ones(lenNodesPlane0))
                
                nodesPlane1  = fem.GetNodesInPlane(fb[i]['p1'], fb[i]['axis1'])

                # superelement
                centerPointAverage0 = fem.GetNodePositionsMean(nodesPlane0)
                centerPointAverage1 = fem.GetNodePositionsMean(nodesPlane1)


                # print('body'+str(i)+'nodes1=', nodesPlane1)
                lenNodesPlane1 = len(nodesPlane1)
                weightsPlane1 = np.array((1./lenNodesPlane1)*np.ones(lenNodesPlane1))
                
                boundaryList = [nodesPlane0, nodesPlane1] 
                mPlane0 = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=objFFRF['oFFRFreducedOrder'], 
                                                      meshNodeNumbers=np.array(nodesPlane0), #these are the meshNodeNumbers
                                                      weightingFactors=weightsPlane0, 
                                                      offset=-centerPointAverage0 - [0,0,Lbase]))
                
                mPlane1 = self.mbs.AddMarker(MarkerSuperElementRigid(bodyNumber=objFFRF['oFFRFreducedOrder'], 
                                                              meshNodeNumbers=np.array(nodesPlane1), #these are the meshNodeNumbers
                                                              weightingFactors=weightsPlane1, 
                                                              offset=-centerPointAverage1))
                
                if i==0:
                    baseMarker = mPlane1
                    oGround = self.mbs.AddObject(ObjectGround(referencePosition= [0,0,0]))
                    mGround = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition=fb[i]['p0']))
                    self.mbs.AddObject(GenericJoint(markerNumbers=[mGround, mPlane0], 
                                               constrainedAxes = [1,1,1,1,1,1],
                                               visualization=VGenericJoint(axesRadius=rFlange*0.5, axesLength=rFlange)))
                    
            self.SC.visualizationSettings.contour.outputVariable = exu.OutputVariableType.DisplacementLocal
            self.SC.visualizationSettings.contour.outputVariableComponent = -1 
        else:
            pass
            
            #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++
            print("create CMS element ...")
            
            
            
            #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++
            #animate modes
            if False:
                from exudyn.interactive import AnimateModes
                self.mbs.Assemble()
                self.SC.visualizationSettings.nodes.show = False
                self.SC.visualizationSettings.openGL.showFaceEdges = True
                self.SC.visualizationSettings.openGL.multiSampling=4
                #SC.visualizationSettings.window.renderWindowSize = [1600,1080]
                # SC.visualizationSettings.contour.outputVariable = exu.OutputVariableType.DisplacementLocal
                # SC.visualizationSettings.contour.outputVariableComponent = 0 #component
                
                
                #%%+++++++++++++++++++++++++++++++++++++++
                #animate modes of ObjectFFRFreducedOrder (only needs generic node containing modal coordinates)
                self.SC.visualizationSettings.general.autoFitScene = False #otherwise, model may be difficult to be moved
                
                nodeNumber = objFFRF['nGenericODE2'] #this is the node with the generalized coordinates
                AnimateModes(self.SC, self.mbs, nodeNumber)
                import sys
                sys.exit()
        
        #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++
        #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++
        #robotics part
        graphicsBaseList = []
        if not useFlexBody:
            #graphicsBaseList +=[graphics.Brick([0,0,-0.15], [0.12,0.12,0.1], graphics.color.grey)]
        
            graphicsBaseList +=[graphics.Cylinder([0,0,-Lbase], [0,0,Lbase-flangeBaseL], rBase, graphics.color.blue)]
            graphicsBaseList +=[graphics.Cylinder([0,0,-flangeBaseL], [0,0,flangeBaseL], flangeBaseR, graphics.color.blue)]
            graphicsBaseList +=[graphics.Cylinder([0,0,0], [0.25,0,0], 0.00125, graphics.color.red)]
            graphicsBaseList +=[graphics.Cylinder([0,0,0], [0,0.25,0], 0.00125, graphics.color.green)]
            graphicsBaseList +=[graphics.Cylinder([0,0,0], [0,0,0.25], 0.00125, graphics.color.blue)]
        
        #base graphics is fixed to ground!!!
        graphicsBaseList +=[graphics.CheckerBoard([0,0,-Lbase], size=2.5)]
        #newRobot.base.visualization['graphicsData']=graphicsBaseList
        
        ty = 0.03
        tz = 0.04
        zOff = -0.05
        toolSize= [0.05,0.5*ty,0.06]
        graphicsToolList = [graphics.Cylinder(pAxis=[0,0,zOff], vAxis= [0,0,tz], radius=ty*1.5, color=graphics.color.red)]
        graphicsToolList+= [graphics.Brick([0,ty,1.5*tz+zOff], toolSize, graphics.color.grey)]
        graphicsToolList+= [graphics.Brick([0,-ty,1.5*tz+zOff], toolSize, graphics.color.grey)]
        if self.computedTorqueControl:
            import roboticstoolbox as rtb
            self.rtbRobot = rtb.models.DH.Puma560()
            self.rtbRobot.links[0].d = L[0]
            # self.rtbRobot.tool.t = [0,0,0.1]#(self, 0.01)
        #configurations and trajectory
        if flagComputeModel: 
            # [u,v,a] = self.mbs.variables['Trajectory'].Evaluate(t)
            pass
            # q0 = u.tolist()
        else: 
            
            # q0 = [0,0,0,0,0,0] #zero angle configuration
            
            #this set of coordinates only works with TSD, not with old fashion load control:
            # q1 = [0, pi/8, pi*0.75, 0,pi/8,0] #configuration 1
            # q2 = [pi,-pi, -pi*0.5,1.5*pi,-pi*2,pi*2] #configuration 2
            # q3 = [3*pi,0,-0.25*pi,0,0,0] #zero angle configuration
            
            #this set also works with load control:
            q1 = [0, pi/8, pi*0.5, 0,pi/8,0] #configuration 1
            # q1 = [0]*6
            # q2 = [0.8*pi,0.5*pi, -pi*0.5,0.75*pi,-pi*0.4,pi*0.4] #configuration 2
            # q3 = [0.5*pi,0,-0.25*pi,0,0,0] #zero angle configuration
            q2 = [0.8*pi] + q1[1:]
            q3 = [0.8*pi, 0] + q1[2:]
            #trajectory generated with optimal acceleration profiles:
            trajectory = Trajectory(initialCoordinates=q0, initialTime=0)
            trajectory.Add(ProfileConstantAcceleration(q3,0.4))
            trajectory.Add(ProfileConstantAcceleration(q1,0.4))
            trajectory.Add(ProfileConstantAcceleration(q2,0.4))
            trajectory.Add(ProfileConstantAcceleration(q0,0.4))
            #traj.Add(ProfilePTP([1,1],syncAccTimes=False, maxVelocities=[1,1], maxAccelerations=[5,5]))
            
            # x = traj.EvaluateCoordinate(t,0)
        
        
        #changed to new robot structure July 2021:
        newRobot = Robot(gravity=gravity,
                      base = RobotBase(visualization=VRobotBase(graphicsData=graphicsBaseList)),
                      tool = RobotTool(HT=HTtranslate([0,0,0.1]), visualization=VRobotTool(graphicsData=graphicsToolList)),
                      referenceConfiguration = q0) #referenceConfiguration created with 0s automatically
        
        fInertia = 1
        #modKKDH according to Khalil and Kleinfinger, 1986
        link0={'stdDH':[0,L[0],0,pi/2], 
               'modKKDH':[0,0,0,0], 
                'mass':20,  #not needed!
                # 'inertia':np.diag([1e-8*0,0.35,1e-8*0]), #w.r.t. COM! in stdDH link frame
                'inertia':np.diag([0.2,0.35,0.2]), #w.r.t. COM! in stdDH link frame; changed because the values where unphysical! 
                'COM':[0,0,0]} #in stdDH link frame
        
        link1={'stdDH':[0,0,L[1],0],
               'modKKDH':[0.5*pi,0,0,0], 
                'mass':17.4, 
                'inertia':np.diag([0.13,0.524,0.539]), #w.r.t. COM! in stdDH link frame
                'COM':[-0.3638, 0.006, 0.2275]} #in stdDH link frame
        
        link2={'stdDH':[0,L[2],W[2],-pi/2], 
               'modKKDH':[0,0.4318,0,0.15], 
                'mass':4.8, 
                # 'inertia':np.diag([0.066,0.086,0.0125]), #w.r.t. COM! in stdDH link frame; unphysical!
                'inertia':np.diag([0.066,0.086,0.025])*fInertia,  # last value changed because it is unphysical!
                'COM':[-0.0203,-0.0141,0.07]} #in stdDH link frame
        
        link3={'stdDH':[0,L[3],0,pi/2], 
               'modKKDH':[-0.5*pi,0.0203,0,0.4318], 
                'mass':0.82, 
                'inertia':np.diag([0.0018,0.0013,0.0018]), #w.r.t. COM! in stdDH link frame
                'COM':[0,0.019,0]} #in stdDH link frame
        
        link4={'stdDH':[0,0,0,-pi/2], 
               'modKKDH':[0.5*pi,0,0,0], 
                'mass':0.34, 
                'inertia':np.diag([0.0003,0.0004,0.0003]), #w.r.t. COM! in stdDH link frame
                'COM':[0,0,0]} #in stdDH link frame
        
        link5={'stdDH':[0,0,0,0], 
               'modKKDH':[-0.5*pi,0,0,0], 
                'mass':0.09, 
                'inertia':np.diag([0.00015,0.00015,4e-5]), #w.r.t. COM! in stdDH link frame
                'COM':[0,0,0.032]} #in stdDH link frame
        linkList=[link0, link1, link2, link3, link4, link5]
        
        #control parameters, per joint:
        df = 1
        Pcontrol = np.array([40000, 40000, 40000*0.25, 100, 100, 10])
        Dcontrol = np.array([400,   400,   400,   1*df,   1*df,   0.1*df])
        
        for i, link in enumerate(linkList):
            newRobot.AddLink(RobotLink(mass=link['mass'], 
                                       COM=link['COM'], 
                                       inertia=link['inertia'], 
                                       localHT=StdDH2HT(link['stdDH']),
                                       PDcontrol=(Pcontrol[i], Dcontrol[i]), 
                                       ))
        
        from exudyn.robotics.models import ManipulatorPuma560, LinkDict2Robot
        
        showCOM = False
        for cnt, link in enumerate(newRobot.links):
            color = graphics.colorList[cnt]
            color[3] = 0.75 #make transparent
            link.visualization = VRobotLink(jointRadius=0.055, jointWidth=0.055*2, showMBSjoint=False,
                                            linkWidth=2*0.05, linkColor=color, showCOM= showCOM )
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #test robot model
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #desired angles:
        qE = q0
        qE = [pi*0.5,-pi*0.25,pi*0.75, 0,0,0]
        tStart = [0,0,0, 0,0,0]
        duration = 0.1
        
        
        jointList = [0]*newRobot.NumberOfLinks() #this list must be filled afterwards with the joint numbers in the mbs!
        
        def ComputeMBSstaticRobotTorques(newRobot, mbs):
            q=[]
            for joint in jointList:
                q += [mbs.GetObjectOutput(joint, exu.OutputVariableType.Rotation)[2]] #z-rotation
            HT=newRobot.JointHT(q)
            return newRobot.StaticTorques(HT)
        
        #++++++++++++++++++++++++++++++++++++++++++++++++
        #base, graphics, object and marker:
        
        objectGround = self.mbs.AddObject(ObjectGround(referencePosition=HT2translation(newRobot.GetBaseHT()), 
                                              #visualization=VObjectGround(graphicsData=graphicsBaseList)
                                                  ))
        
        if not useFlexBody:
            baseMarker = self.mbs.AddMarker(MarkerBodyRigid(bodyNumber=objectGround, localPosition=[0,0,0]))
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #build mbs robot model:
        robotDict = newRobot.CreateRedundantCoordinateMBS(self.mbs, baseMarker=baseMarker)
            
        jointList = robotDict['jointList'] #must be stored there for the load user function
        unitTorques0 = robotDict['unitTorque0List'] #(left body)
        unitTorques1 = robotDict['unitTorque1List'] #(right body)
        loadList0 = robotDict['jointTorque0List'] #(left body)
        loadList1 = robotDict['jointTorque1List'] #(right body)
        #print(loadList0, loadList1)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #control robot
        compensateStaticTorques = False
        torsionalSDlist = robotDict['springDamperList']
        self.mbs.variables['q0'] = q0
        #user function which is called only once per step, speeds up simulation drastically
        self.mbs.variables['debugData'] = []
        self.mbs.variables['computedTorqueControl'] =  self.computedTorqueControl
        def PreStepUF(mbs, t):
            if 'Trajectory' in mbs.variables.keys(): 
                [u,v,a] = mbs.variables['Trajectory'].Evaluate(t)
            else: 
                # [u,v,a] = [[0]*6, [0]*6, [0]*6] # trajectory.Evaluate(t)
                u = mbs.variables['q0']
                v = [0]*6
                a = [0]*6                
            if mbs.variables['computedTorqueControl']: 
                controlTorques = self.rtbRobot.rne(u, v, a, gravity=[0,0,-9.81])
            elif compensateStaticTorques:
                controlTorques  = ComputeMBSstaticRobotTorques(newRobot, mbs)
            
            else:
                controlTorques  = np.zeros(len(jointList))
                
            
            #compute load for joint number
            mbs.variables['debugData'] += [[t]]
            
            for i in range(len(jointList)):
                joint = jointList[i]
                phi = mbs.GetObjectOutput(joint, exu.OutputVariableType.Rotation)[2] #z-rotation
                omega = mbs.GetObjectOutput(joint, exu.OutputVariableType.AngularVelocityLocal)[2] #z-angular velocity
                tsd = torsionalSDlist[i]
                mbs.SetObjectParameter(tsd, 'offset', u[i])
                mbs.SetObjectParameter(tsd, 'velocityOffset', v[i])
                mbs.SetObjectParameter(tsd, 'torque', controlTorques[i]) #additional torque from given control strategy 
                mbs.variables['debugData'][-1] += [u[i], phi]
            return True
        
        self.mbs.SetPreStepUserFunction(PreStepUF)
        
        
        
        if useFlexBody:
            baseType = 'Flexible'
        else:
            baseType = 'Rigid'
        
        #add sensors:
        cnt = 0
        self.jointTorque0List = []
        self.jointRotList = []
        for i in range(len(jointList)):
            jointLink = jointList[i]
            tsd = torsionalSDlist[i]
            #using TSD:
            sJointRot = self.mbs.AddSensor(SensorObject(objectNumber=tsd, 
                                       fileName='solution/joint' + str(i) + 'Rot'+baseType+'.txt',
                                       outputVariableType=exu.OutputVariableType.Rotation,
                                       storeInternal=True, 
                                       writeToFile = sensorWriteToFile))
            self.jointRotList += [sJointRot]
        
            sJointAngVel = self.mbs.AddSensor(SensorObject(objectNumber=jointLink, 
                                       fileName='solution/joint' + str(i) + 'AngVel'+baseType+'.txt',
                                       outputVariableType=exu.OutputVariableType.AngularVelocityLocal,
                                       writeToFile = sensorWriteToFile))
        
            sTorque = self.mbs.AddSensor(SensorObject(objectNumber=tsd, 
                                    fileName='solution/joint' + str(i) + 'Torque'+baseType+'.txt',
                                    outputVariableType=exu.OutputVariableType.TorqueLocal,
                                    writeToFile = sensorWriteToFile))
        
            
        
            self.jointTorque0List += [sTorque]
        
        if self.outputType in [0,1,2,3]: 
            self.sHandPos = self.mbs.AddSensor(SensorBody(bodyNumber=robotDict['bodyList'][-1], 
                                    fileName='solution/handPos'+baseType+'.txt',
                                    outputVariableType=exu.OutputVariableType.Position, storeInternal=True, 
                                    writeToFile = sensorWriteToFile))
        
            # sHandVel = self.mbs.AddSensor(SensorBody(bodyNumber=robotDict['bodyList'][-1], 
            #                         fileName='solution/handVel'+baseType+'.txt' storeInternal=True, 
            #                         outputVariableType=exu.OutputVariableType.Velocity,
            #                         writeToFile = sensorWriteToFile))
    
            self.sHandRot = self.mbs.AddSensor(SensorBody(bodyNumber=robotDict['bodyList'][-1], 
                                    fileName='solution/handRot'+baseType+'.txt',
                                    outputVariableType=exu.OutputVariableType.Rotation,
                                    writeToFile = sensorWriteToFile, storeInternal=True))
        if self.outputType in [2, 3]: 
            self.sFlangePos = self.mbs.AddSensor(SensorMarker(markerNumber=baseMarker,
            # self.sFlangePos = self.mbs.AddSensor(SensorSuperElement(bodyNumer=objFFRF['oFFRFreducedOrder'], 
            #                                    meshNodeNumbers=nodesArmStartPlane, 
                                                outputVariableType = exu.OutputVariableType.Position,  
                                                storeInternal=True))
            
            self.sFlangeRot = self.mbs.AddSensor(SensorMarker(markerNumber=baseMarker,
                                                         outputVariableType = exu.OutputVariableType.Rotation,  
                                                         storeInternal=True))       

            # self.mbs.AddLoad(Force(markerNumber=baseMarker, bodyFixed = True, loadVector=[100*0,0,0]))       # Force only for debugging      
        
        self.mbs.Assemble()
        #mbs.systemData.Info()
        PreStepUF(self.mbs, 0)
        
        self.SC.visualizationSettings.connectors.showJointAxes = True
        self.SC.visualizationSettings.connectors.jointAxesLength = 0.02
        self.SC.visualizationSettings.connectors.jointAxesRadius = 0.002
        
        self.SC.visualizationSettings.nodes.show = False
        # SC.visualizationSettings.nodes.showBasis = True
        # SC.visualizationSettings.nodes.basisSize = 0.1
        self.SC.visualizationSettings.loads.show = False
        
        self.SC.visualizationSettings.openGL.multiSampling=4
            
           
        # 
        #mbs.WaitForUserToContinue()
        self.simulationSettings = exu.SimulationSettings() #takes currently set values or default values
        self.simulationSettings.timeIntegration.numberOfSteps = self.GetNSimulationSteps()
        self.simulationSettings.timeIntegration.endTime = self.endTime
        self.simulationSettings.solutionSettings.solutionWritePeriod = self.endTime/self.nStepsTotal
        self.simulationSettings.solutionSettings.sensorsWritePeriod = self.endTime/self.nStepsTotal
        self.simulationSettings.solutionSettings.binarySolutionFile = True
        self.simulationSettings.solutionSettings.writeSolutionToFile = False
        self.simulationSettings.timeIntegration.verboseMode = 0
        # simulationSettings.displayComputationTime = True
        self.simulationSettings.displayStatistics = True
        self.simulationSettings.linearSolverType = exu.LinearSolverType.EigenSparse
        
        #simulationSettings.timeIntegration.newton.useModifiedNewton = True
        self.simulationSettings.timeIntegration.generalizedAlpha.useIndex2Constraints = True
        self.simulationSettings.timeIntegration.generalizedAlpha.useNewmark = self.simulationSettings.timeIntegration.generalizedAlpha.useIndex2Constraints
        self.simulationSettings.timeIntegration.newton.useModifiedNewton = True
        
        self.simulationSettings.timeIntegration.generalizedAlpha.computeInitialAccelerations = True
        self.SC.visualizationSettings.general.autoFitScene=False
        self.SC.visualizationSettings.window.renderWindowSize=[1200,1200]
        self.SC.visualizationSettings.openGL.shadow = 0.25
        self.SC.visualizationSettings.openGL.light0position = [-2,5,10,0]
        
        
        self.simulationSettings.staticSolver.verboseMode = 0
        if self.verboseMode: 
            self.simulationSettings.displayStatistics = True
        
        useGraphics = False # only for testing
        if self.computeStatic: 
            self.mbs.SolveStatic(self.simulationSettings, updateInitialValues=True)
            # print('computedStatic')
            
            
        if useGraphics:
            exu.StartRenderer()
            if 'renderState' in exu.sys:
                self.SC.SetRenderState(exu.sys['renderState'])
            self.mbs.WaitForUserToContinue()
            self.mbs.SolveDynamic(self.simulationSettings, showHints=True)
            
        return

    #get time vector according to output data
    def GetOutputXAxisVector(self):
        return self.timeVecOut

    #create a randomized input vector
    #relCnt can be used to create different kinds of input vectors (sinoid, noise, ...)
    #isTest is True in case of test data creation
    def CreateInputVector(self, relCnt = 0, isTest=False, dataErrorEstimator = False):
        def SetRandomizedZero(p0, qi, qj):
            if p0 > 1.0 or p0 < 0.0: # 
                raise ValueError("probability must be between 0 and 1")
            if np.random.random() > p0: # e.g. 20 percent chance to set a random number of axes to zero
                nAxs = int(np.ceil(np.random.random() * 6))
                iAxs = random.sample([0,1,2,3,4,5], nAxs) # choose which axes to set to zero 
                iAxs = np.sort(iAxs)
                for i in iAxs: 
                    qi[i] = qj[i]
            return qi
            
        vec = np.zeros(self.GetInputScaling().shape)
        qEnd = 2*np.random.rand(self.nJoints)-1
        myTraj = None
        
        if self.inputType == 0: 
            vec = endAngles
        elif self.inputType == 1: 
            qStart = np.random.uniform(self.qLim[0], self.qLim[1]) / np.pi
            vec = np.array([qStart, qEnd]).T
        elif self.inputType == 2: 
            qStart = np.random.uniform(self.qLim[0], self.qLim[1]) / np.pi
            myTraj = Trajectory(qStart)
            myTraj.Add(ProfileConstantAcceleration(qEnd,0.5))
                
        elif self.inputType == 3: 
            qStart = np.random.uniform(self.qLim[0], self.qLim[1])
            qEnd =  np.random.uniform(self.qLim[0], self.qLim[1])
            myTraj = Trajectory(qStart)
            myTraj.Add(ProfilePTP(qEnd, maxVelocities =  self.vMax, maxAccelerations=  self.aMax, syncAccTimes=False))

        elif self.inputType == 4: 
            q1 =  np.random.uniform(self.qLim[0], self.qLim[1])
            q2 =  np.random.uniform(self.qLim[0], self.qLim[1])
            q3 =  np.random.uniform(self.qLim[0], self.qLim[1])
            if np.random.random() > 0.8: # 20 percent chance  to set a random number of axes to zero
                nAxs = int(np.ceil(np.random.random() * 6))
                iAxs = random.sample([0,1,2,3,4,5], nAxs) # choose which axes to set to zero
                iAxs = np.sort(iAxs)
                flag_a = np.random.random() > 0.75
                for i in iAxs: 
                    if flag_a: 
                        q2[i] = q1[i]
                    else: 
                        q3[i] = q2[i]
                    
            myTraj = Trajectory(q1)
            myTraj.Add(ProfileConstantAcceleration(q2, self.endTime/2))
            myTraj.Add(ProfileConstantAcceleration(q3, self.endTime/2))
        
        elif self.inputType == 5: 
            qStart = np.random.uniform(self.qLim[0], self.qLim[1])
            myTraj = Trajectory(qStart)
            tTraj = 0
            while tTraj < self.timeVecIn[-1]: 
                qNew = np.random.uniform(self.qLim[0], self.qLim[1])
                qNew = SetRandomizedZero(0.8, qNew, myTraj.GetFinalCoordinates())
                if dataErrorEstimator: 
                    myTraj.Add(ProfilePTP(qNew, syncAccTimes=False, maxVelocities = self.vMax*2, maxAccelerations=self.aMax*2))
                else: 
                    myTraj.Add(ProfilePTP(qNew, syncAccTimes=False, maxVelocities = self.vMax, maxAccelerations=self.aMax))
                tTraj = myTraj.GetTimes()[-1]
                
            
        if self.inputType in [2,3,4,5]: 
            vec = np.zeros([self.nStepsTotal, self.nJoints])
            for i, t in enumerate(self.timeVecIn): 
                vec[i,:], _, _ = myTraj.Evaluate(t)
                vec[i,:] /= self.GetInputScaling()[0,:]
                
        else: 
            raise ValueError('For 6R flexible robot input type {} is not defined.'.format(self.inputType))
            
        # attention: here the vector AND the trajectory object are passed; this is done for convenience; the simulation model directly 
        # evalutes the "myTraj" object to obtain smooth angles. 
        return vec, myTraj 
            
    #get number of simulation steps
    def GetNSimulationSteps(self):
        return self.nStepsTotal*20 # x finer simulation than output to ensure convergence.

    #split input data into initial values, forces or other inputs
    #return dict with 'data' and possibly 'initialODE2' and 'initialODE2_t'
    def SplitInputData(self, inputData):
        rv = {}
        if self.inputType == 0:
            rv['qEnd'] = inputData    
            
        elif self.inputType == 1: 
            rv['qStart'] = inputData[:,0]
            rv['qEnd'] = inputData[:,1]

        elif self.inputType in [2,3,4,5]: 
            rv['q'] = inputData[:,0:self.nJoints] 
            
        else: 
            data = np.zeros((self.nStepsTotal, self.nJoints+1))
            data[:,0] = self.timeVecIn
    
            for j in range(self.nJoints):
                if not self.IsFFN():
                    data[:,j+1] = inputData[:,j]
                else:
                    data[:,j+1] = inputData[j*self.nStepsTotal:(j+1)*self.nStepsTotal]
            
            rv['data'] = data
        return rv

    #split output data to get ODE2 values (and possibly other data, such as ODE2)
    #return dict {'ODE2':[], 'ODE2_t':[]}
    def SplitOutputData(self, outputData):
        rv = {}

        data = outputData
        if self.outputType == 0 and False: 
            print('warning: splitOutputData for this outputType ({}) not implemented'.format(outputType))
            
        else: 
            if outputData.ndim == 1:
                data = outputData.reshape((self.nStepsTotal,3*self.nOutputPos))
            
            coords = ['X','Y','Z']
    
            for i, c in enumerate(coords):
                rv['posArm'+c] = data[:, i]
    
            if self.velocityGround:
                for i, c in enumerate(coords):
                    rv['velGround'+c] = data[:, 3+i]

        return rv
    
    #convert all output vectors into plottable data (e.g. [time, x, y])
    #the size of data allows to decide how many columns exist
    def OutputData2PlotData(self, outputData, forSolutionViewer=False):
        timeVec = self.GetOutputXAxisVector()
        dataDict = self.SplitOutputData(outputData)

        if self.velocityGround:
            data = np.vstack((timeVec, 
                              dataDict['posArmX'].T, dataDict['posArmY'].T, dataDict['posArmZ'].T,
                              dataDict['velGroundX'].T, dataDict['velGroundY'].T, dataDict['velGroundZ'].T, 
                              )).T
        else:
            data = np.vstack((timeVec, 
                              dataDict['posArmX'].T, dataDict['posArmY'].T, dataDict['posArmZ'].T,
                              )).T
        return data

    #return dict of names to columns for plotdata        
    def PlotDataColumns(self):
        d = {'time':0}
        coords = ['X','Y','Z']
        for i, c in enumerate(coords):
            d['posArm'+c] = i+1
            if self.velocityGround:
                d['velGround'+c] = i+1+3
            
        return d

    #get compute model with given input data and return output data
    #initialState contains position and velocity states as list of two np.arrays 
    def ComputeModel(self, inputData, hiddenData=None, verboseMode = 0, solutionViewer = False, flagRenderer=False, myTraj = None, computeEigenValues = False):
        #set input data ...
        if type(inputData) is tuple: 
            inputData, myTraj = inputData
            
        inputDict = self.SplitInputData(self.GetInputScaling() * np.array(inputData))
        if self.inputType == 0: 
            # self.mbs.variables['jointAngles'] = inputDict['qEnd']
            
            # q1 = [0, pi/8, pi*0.5, 0,pi/8,0] #configuration 1
            q1 = [0, 0, 0, 0, 0, 0]
            q2  = inputDict['qEnd']
            # q2 = self.mbs.variables['jointAngles']
            if self.verboseMode: 
                print(q2)
            
            self.CreateModel(q0 = q1, flagComputeModel = True)
            self.mbs.variables['Trajectory'] = Trajectory(initialCoordinates=q1, initialTime=0)
            self.mbs.variables['Trajectory'].Add(ProfileConstantAcceleration(q2,0.5))
            
        elif self.inputType in [1, 2]: 
            if self.inputType == 1: 
                q1 = inputDict['qStart']
                q2  = inputDict['qEnd']
                tMotion = 0.5
            else: 
                q1 = inputDict['q'][0,:]
                # getend coordinates
                for i, q in enumerate(inputDict['q'], start=5): 
                    if np.linalg.norm(inputDict['q'][i-1] -  inputDict['q'][i]) < 1e-12: 
                        q2 = inputDict['q'][i-1]
                        nMotion = i-1
                        tMotion = nMotion* np.mean(np.diff(self.timeVecIn))
                        break
            if self.verboseMode: print("\nq1: {}\nq2: {}".format(q1, q2))
            self.CreateModel(q0 = q1, flagComputeModel = True, computeStatic = self.computeStatic)
            self.mbs.variables['Trajectory'] = Trajectory(initialCoordinates=q1, initialTime=0)
            self.mbs.variables['Trajectory'].Add(ProfileConstantAcceleration(q2,tMotion))
        
        elif self.inputType == 3: 
            # print('Trajectory: ', myTraj)
            self.CreateModel(q0 = inputDict['q'][0,:], flagComputeModel = True)
            self.mbs.variables['Trajectory'] = myTraj

        elif self.inputType == 4: 
            self.CreateModel(q0 = inputDict['q'][0,:], flagComputeModel = True)
            self.mbs.variables['Trajectory'] = myTraj
 
        elif self.inputType == 5: 
                self.CreateModel(q0 = inputDict['q'][0,:], flagComputeModel = True)
                self.mbs.variables['Trajectory'] = myTraj

            
            
        else: 
            self.mbs.variables['jointAngles'] = inputDict['q']
            self.CreateModel()


        self.simulationSettings.timeIntegration.verboseMode = self.verboseMode
        self.simulationSettings.solutionSettings.writeSolutionToFile = solutionViewer

        if flagRenderer: 
            self.SC.visualizationSettings.openGL.initialModelRotation = RotationMatrixZ(np.pi/5) @ RotXYZ2RotationMatrix([1.2*np.pi/3,0,0])
            exu.StartRenderer()
            self.mbs.WaitForUserToContinue()
        if computeEigenValues: 
            self.simulationSettings.solutionSettings.writeSolutionToFile = True
            
        self.mbs.SolveDynamic(self.simulationSettings, solverType = exu.DynamicSolverType.GeneralizedAlpha)
        
        if solutionViewer:
            self.mbs.SolutionViewer()

        #get sensor data and apply piecewise scaling:
        output = 0*self.GetOutputScaling()
        if self.outputType == 0 or self.outputType == 1: 
            output[:,0:3] = self.mbs.GetSensorStoredData(self.sHandPos)[1:,1:]
        if self.outputType == 1: 
            output[:,3:6] = self.mbs.GetSensorStoredData(self.sHandRot)[1:,1:]
        if False: 
            for i, sensor in enumerate(self.listSensors):
                #output[3*self.nStepsTotal*i:3*self.nStepsTotal*(i+1)] = self.mbs.GetSensorStoredData(sensor)[:,1:].T.flatten()
                # output[:,3*i:3*(1+i)] = self.mbs.GetSensorStoredData(sensor)[:,1:]
                # output 
                pass
            
        if self.outputType in [2,3]: 
            # note that the flange is approximatly at [0,0,0] --> the ground is lower
            # joint 0 is directly connected to the baseMarker (should be Marker Nr. 2)
            posHand_0 = self.mbs.GetSensorStoredData(self.sHandPos)[1:,1:]
            rotHand_0 = self.mbs.GetSensorStoredData(self.sHandRot)[1:,1:]
            posFlange_0 = self.mbs.GetSensorStoredData(self.sFlangePos)[1:,1:]
            rotFlange_0 = self.mbs.GetSensorStoredData(self.sFlangeRot)[1:,1:]
            # transform positions relative to flange
            flagDebug = False
            if flagDebug: 
                import roboticstoolbox as rtb
                L = [0.075,0.4318,0.15,0.4318]
                robot = rtb.models.DH.Puma560()
                robot.links[0].d = L[0]
                robot.tool.t = [0,0,0.1*0]
                q =  []
                for i in range(6): 
                    q += [self.mbs.GetSensorStoredData(self.jointRotList[i])[1:,1:].reshape(-1)]
                q = np.array(q).T
            

                    
            for i in range(self.nStepsTotal-self.nOutputSteps, self.nStepsTotal): 
                iOut = i -self.nOutputSteps
                RFlange = RotXYZ2RotationMatrix(rotFlange_0[i,:])
                REE = RotXYZ2RotationMatrix(rotHand_0[i,:])
                
                R_flange_EE = np.linalg.inv(RFlange) @ REE # transform from EE to 0; is equal to forward kinematics Orientation of tool
                
                pos_flange_EE = np.linalg.inv(RFlange) @ (posHand_0[i,:] - posFlange_0[i,:]) # transform from global into 
                
                
                output[iOut,0:3] = posHand_0[i,:] - pos_flange_EE

                if self.outputType == 3: # if outputtype 3: fill also orientation in
                    diffRot = REE.T @ R_flange_EE
                    ang = RotationMatrix2RotXYZ(diffRot)
                    rotVec = RotationMatrix2RotationVector(diffRot) # for small values very close! 
                
                    output[iOut,3:7] = rotVec
                    # note: maybe unity vectors have better properties than?
                    
                    
                # test: fkine von Roboter: 
                if flagDebug: 
                    # qEnd = [dataEnd[1], dataEnd[3], dataEnd[5], dataEnd[7], dataEnd[9], dataEnd[11]]
                    T_ = robot.fkine(q[i,:])
                    posFkine = T_.t
                    rotFkine = T_.R
                    diffPos = pos_flange_EE - posFkine
                    diffRot =  RotationMatrix2RotXYZ(R_flange_EE.T @ rotFkine)
                    # dataPrint = SmartRound2String(, 3)
                    print('Check diff fkine vs sensors: ', SmartRound2StringVector(diffPos, 3), " | ", 
                                                           SmartRound2StringVector(diffRot, 3))
                    # print('Check:angle rot_flange_EE - fkine: ', SmartRound2String(diffPos, 3))
        
            # for i in range(self.nStepsTotal):       
            #     RFlange = RotXYZ2RotationMatrix(rotFlange_0[i,:])
            #     REE = RotXYZ2RotationMatrix(rotHand_0[i,:])
            #     R_flange_EE = np.linalg.inv(RFlange) @ REE 
                
            
            
            
        if not(self.nOutputSteps is None) and not (self.nOutputSteps == self.nStepsTotal): 
            output = output[-self.nOutputSteps:,:] 

        output = self.GetOutputScaling()*output
        return output
   
        
        
        
    #convert all output vectors into plottable data (e.g. [time, x, y])
    #the size of data allows to decide how many columns exist
    def OutputData2PlotData(self, outputData, forSolutionViewer=False):

        
        nDiff = self.nStepsTotal - self.nOutputSteps
        leadingZeros = np.zeros([nDiff, len(self.GetOutputScaling()[1])])
       
        timeVec = self.GetOutputXAxisVector()
        outData = np.vstack((leadingZeros, outputData ))
        
        data = np.vstack((timeVec, outData.T)).T
            
        return data
    


#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__': #include this to enable parallel processing
    # test_init()    
    import matplotlib.pyplot as plt
    np.random.seed(42)

    tscale = 1
    nOut = 200
    flagCompare = False
    
    if not(flagCompare): 
        modelSCFlexible = SliderCrank(nStepsTotal=200, nOutputSteps = nOut, tStartup = 0.5, endTime= 3, useInitialVelocities=False, 
                                      useInitialAngles=True, useTorqueInput=False, flagFlexible=True, useVelocityInput = False,
                                          usePosInput = True, flagVelNoise=False, trajType = 0, outputType = 1, 
                                               initAngleRange = [-np.pi,np.pi], # initVelRange  = [-12,12],
                                               vMax = 5, aMax = 20)
        
        
    
    
    else: 
        modelSC = SliderCrank(nStepsTotal=200, endTime= 2, useInitialVelocities=False, useInitialAngles=True, 
                                          useTorqueInput=False, flagFlexible=False, flagResistanceSlider=True, nOutputSteps=nOut, 
                                          useVelocityInput = False, usePosInput = True, outputType = 0, 
                                                initAngleRange = [-np.pi, np.pi], # initVelRange  = [-12,12],
                                                vMax = 8, aMax = 20)
        
        modelSCFlexible = SliderCrank(nStepsTotal=200, nOutputSteps = nOut, tStartup = 0.5, endTime= 2, useInitialVelocities=False, 
                                      useInitialAngles=True, useTorqueInput=False, flagFlexible=True, useVelocityInput = False,
                                          usePosInput = True, flagVelNoise=False, trajType = 1, outputType = 0, 
                                               initAngleRange = [-np.pi,np.pi], # initVelRange  = [-12,12],
                                               vMax = 8, aMax = 20)
    
        modelSC.CreateModel()
    modelSCFlexible.CreateModel()
    from matplotlib import colors as mColors
    myColors = list(mColors.TABLEAU_COLORS)
    if True: # plot 
        fig, axs = plt.subplots(2)
        xLabel, yLabel = ['', 't in s'], [r'$\varphi$ in rad', r'$\omega$ in rad/s']
        for i in range(2): 
            axs[i].grid(True)
            axs[i].set(xlabel=xLabel[i], ylabel=yLabel[i])
        
    for i in range(5):
        plt.figure('input')
        vec = modelSCFlexible.CreateInputVector()
        if False:  # hold position
            vec = np.array([1 * np.sin(np.linspace(0, 50, vec.size))])
        if False: 
            plt.plot(AccumulateAngle(np.arctan2(vec[0,:], vec[1,:])))
        else: 
            phi = AccumulateAngle(np.arctan2(vec[0,:], vec[1,:]))
            omega = np.diff(phi)/((modelSCFlexible.endTime/modelSCFlexible.nStepsTotal))
            axs[0].plot(modelSCFlexible.timeVecIn[:-34], phi[34:])
            axs[1].plot(modelSCFlexible.timeVecIn[:-35], omega[34:])
            plt.plot([0,2.5], [5]*2, 'k--')
            plt.plot([0,2.5], [5]*2, 'k--')
            continue
        # vec[0,100:] = vec[0,100]
        # vec[1,100:] = vec[1,100]
        # continue
        plt.figure('output')
        flagViewSolution = False
        if i in []: 
            flagViewSolution=True
        if flagCompare: 
            out = modelSC.ComputeModel(vec, verboseMode=0, solutionViewer=flagViewSolution)
            plt.plot(out, label='output Nr. {}'.format(i), color=myColors[i])
            
        
        outFlex = modelSCFlexible.ComputeModel(vec, verboseMode=0, solutionViewer=flagViewSolution)
        plt.plot(outFlex, '--', label='output Nr. {} flexible'.format(i), color=myColors[i])
        
        
        if False: # check control ...  
            data = np.array(modelSCFlexible.debugData)
            plt.figure('Control')
            plt.plot(data[:,0], 'g-', label='phi desired')
            plt.plot(data[:,1], 'r-', label='phi current')
            # plt.plot(data[:,2], 'g--', label='v desired')
            # plt.plot(data[:,3], 'r--', label='v current')
            plt.legend()
            plt.grid(True)
        # elif: 
            # plt.figure('input')    
            
        else: 
            data = np.array(modelSCFlexible.debugData)
            plt.figure('Control')
            plt.plot(data[:,0], 'g-', label='phi desired')
            plt.plot(data[:,1], 'r-', label='phi current')
            plt.legend()
    axs[1].plot([0,2.5], [5]*2, 'k--')
    axs[1].plot([0,2.5], [-5]*2, 'k--')
    
    sys.exit()
    plt.figure('output')
    plt.ylabel(r'slider $x_{pos}$ in m')
    plt.xlabel('datapoint number')
    plt.title(r'$\tau_{in}$: sin/cos, $\varphi_0 \in $ [-\pi,\pi], $\dot{\varphi}_0 \in [-8, 8]$')
    plt.legend()
    plt.grid()
    
    
        
    # plt.figure()
    # plt.plot(vec[:,2:].T)
    sys.exit()
    
    
    #after 2 seconds, oscillations are gone
    # model = NonlinearOscillatorContinuous(nStepsTotal=200, endTime=2,
    #                                       frictionForce=1*0,
    #                                       initUfact = 0.1, initVfact = 0.5)
    model = NonlinearOscillator(useVelocities=False, useInitialValues=True,
                                nStepsTotal=63, endTime=1, variationMKD=False,
                                nMasses=1, useFriction=False, 
                                )
    
    model.CreateModel()
    for i in range(10): 
    
        vec = model.CreateInputVector()
        out = model.ComputeModel(vec)
        
        inputData = 1*model.inputScaling #don't change input scaling!
        inputData[:,0] = 1.
        inputData[100:,0] = 0.
        #inputData = model.CreateInputVector()
        # inputType can be 'Force' or 'state'
        output = model.ComputeModel(inputData, 
                                    verboseMode=True, solutionViewer=False)
        model.mbs.PlotSensor(0, closeAll=False, newFigure=False)

    if False:
        model = DoublePendulum(nStepsTotal=500, endTime=5, nnType='FFN')

        inputData = [1.6,2.2,0.030,0.330,1.500,2.41] #case2
        output = model.ComputeModel(inputData, 
                                    #hiddenData=[1.6,1.6,0,0],  #for qRNN
                                    verboseMode=True, solutionViewer=False)
        # model.mbs.PlotSensor([model.sAngles]*4,components=[0,1,2,3], closeAll=True,labels=['phi0','phi1','ssst','phi1_t'])
        model.mbs.PlotSensor([model.sAngles]*2,components=[0,2], closeAll=True, labels=['phi0','phi0_t'])
        model.mbs.PlotSensor([model.sPos0,model.sPos1],components=[1,1],componentsX=[0,0], newFigure=True,
                             labels=['m0','m1'])
        print('mbs last step=\n',
              model.mbs.GetSensorStoredData(model.sAngles)[-1,1:])
        
        #%%
        #reference solution for double pendulum with scipy:
        #NOTE theta1 is relative angle!
        import numpy as np
        from scipy.integrate import odeint
        import matplotlib.pyplot as plt
        
        # Constants
        g = 9.81  # gravity
        L0, L1 = inputData[4], inputData[5]  # lengths of arms
        m0, m1 = 2.0, 1.0  # masses
        theta0, theta1 = inputData[0], inputData[1]  # initial angles
        v0, v1 = inputData[2], inputData[3]  # initial angular velocities
        
        # System of differential equations
        def equations(y, t, L0, L1, m0, m1):
            theta0, z0, theta1, z1 = y
            
            c, s = np.cos(theta0-theta1), np.sin(theta0-theta1)
            theta0_dot = z0
            z0_dot = (m1*g*np.sin(theta1)*c - m1*s*(L0*z0**2*c + L1*z1**2) - (m0+m1)*g*np.sin(theta0)) / L0 / (m0 + m1*s**2)
            theta1_dot = z1
            z1_dot = ((m0+m1)*(L0*z0**2*s - g*np.sin(theta1) + g*np.sin(theta0)*c) + m1*L1*z1**2*s*c) / L1 / (m0 + m1*s**2)
            return theta0_dot, z0_dot, theta1_dot, z1_dot
        
        # Initial conditions: theta0, dtheta0/dt, theta1, dtheta1/dt.
        y0 = [theta0, v0, theta1, v1]
        
        # Time array for solution
        t = np.linspace(0, 5, 1000)
        
        # Solve ODE
        solution = odeint(equations, y0, t, args=(L0, L1, m0, m1))
        scipySol = solution[-1,:]
        print('scipy=\n',scipySol[0],scipySol[2],scipySol[1],scipySol[3])
        # Plot
        if False:
            plt.figure(figsize=(10, 4))
            plt.plot(t, solution[:, 0], label="theta0(t)")
            plt.plot(t, solution[:, 2], label="theta1(t)")
            plt.plot(t, solution[:, 1], label="theta0_t(t)")
            plt.plot(t, solution[:, 3], label="theta1_t(t)")
            plt.legend(loc='best')
            plt.xlabel('Time')
            plt.grid()
            plt.show()
    

    
