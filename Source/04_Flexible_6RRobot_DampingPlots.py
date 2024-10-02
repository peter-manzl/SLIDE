#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#           04_Flexible_6RRobot_DampingPlots
# Details:  File for creating plots for damping of the flexible robot model. 
#
# Author:   Peter Manzl, Johannes Gerstmayr
# Date:     2024-10-01
# Copyright: See Licence.txt
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
from simModels import Flex6RRobot
import matplotlib.pyplot as plt
from AISurrogateLib import getDamping, roundSignificant


endTime = 2
nStepsTotal = 128
isRigid = False
nOutput = 64



simModel = Flex6RRobot(nStepsTotal=nStepsTotal, endTime=endTime, 
                 isRigid=isRigid, createModel=False, verboseMode=0, inputType=4, outputType = 2, 
                 nOutputSteps = nOutput, EModulus = 1e9)
simModel.CreateModel([0]*6, flagComputeModel=True)

myTestNumber = 2

if myTestNumber == 1: 
    # running many randomized static tests ...
    t_dList, A_dList, qList = [], [], []
    for i in range(50): 
        q0 = (np.random.random(6)-0.5)*2*np.pi
        qList += [q0]
        print('q0: ', q0)
        simModel.CreateModel(q0)
        simModel.mbs.SolveStatic()
        # simModel.mbs.WaitForUserToContinue()
        t_d, A_d = getDamping(simModel.mbs, 0.01, t=1)
        t_dList += [t_d]
        A_dList += [A_d]
    print('max td: ', np.max(t_dList))

elif myTestNumber == 2: # obtain damping for specific static configurations
    # 
    q0 = [0,  np.pi, -np.pi, -0.38, -2.64, -2.99]
    simModel.CreateModel(q0)
    simModel.mbs.SolveStatic()
    t_d, A_d = getDamping(simModel.mbs, 0.01, t=1)
    
    