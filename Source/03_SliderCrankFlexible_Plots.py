#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#           03_SliderCrankFlexible_Plots
# Details:  File for creating plots for eigenvalues / damping of the flexible slider-crank. 
#
# Author:   Peter Manzl, Johannes Gerstmayr
# Date:     2024-04-08
# Copyright: See Licence.txt
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import matplotlib.pyplot as plt
import numpy as np
from AISurrogateLib import getDamping, roundSignificant
from simModels import SimulationModel, SliderCrank, AccumulateAngle

nStepsTotal = 128
nOut = 32


simModel = SliderCrank(nStepsTotal=nStepsTotal//8, nOutputSteps = nOut//8, tStartup = 0.1, endTime= 0.2, 
                               useInitialVelocities=True, useInitialAngles=True, 
                               useTorqueInput=False, flagFlexible=True,  useVelocityInput = False, 
                               flagVelNoise = True, trajType = 0, usePosInput=True, outputType=1, 
                               initAngleRange = [-np.pi,np.pi],
                               vMax = 8, aMax = 20)
simModel.CreateModel()
# simModel.mbs.Assemble()
t_dList, EV_List, angle = [], [], []
N = 180+1 
# initialize model from 0 to 2 pi (solving statically) and calculate eigenvalues for damping estimation.
for i in range(N): 
    inputData = simModel.CreateInputVector()
    phi = i * (2*np.pi)  / (N-1)
    angle += [phi]
    inputData[0,:] = np.cos(phi)
    inputData[1,:] = np.sin(phi)
    

    output = simModel.ComputeModel(inputData, 
                                verboseMode=False, solutionViewer=False)
    data = getDamping(simModel.mbs, 0.01, nValues=8)
    # ignore two smallest eigenvalues, which correspond to the motion of the crankshaft
    t_dList += [data[0][-3]] 
    EV_List += [data[1]]


font = {'size'   : 18}
plt.rc('font', **font)
EV_List = np.array(EV_List)
plt.semilogy(np.array(angle)*180/np.pi, np.array(EV_List[:,0:-1:2]))

# formatting plot
plt.xticks(np.linspace(0, 360,9)) 
plt.grid()
plt.xlabel(r'$\varphi$ in °')
plt.ylabel('MBS eigenvalues')
plt.legend([r'$v_{}=66.67$'.format(3), r'$v_{} \in [40.3, 48.7]$'.format(2), 
            r'$v_{} \in [2.2, 4.20]$'.format(1), r'$v_{}\approx 0$'.format(0)])


plt.yticks([ 2.5, 5, 10, 20, 40, 80], [2.5,5,10,20,40,80])
plt.ylim([2,80])
print('max td: ', roundSignificant(np.max(t_dList), 3))

#%% 
# visualization of random trajectories
simModelTest = SliderCrank(nStepsTotal=200, nOutputSteps = 200, tStartup = 1, endTime= 9, 
                                  useInitialVelocities=False, useInitialAngles=True, 
                                  useTorqueInput=False, flagFlexible=True,  useVelocityInput = False, 
                                  flagVelNoise = False, trajType = 0, usePosInput=True, outputType=1, 
                                  initAngleRange = [-np.pi,np.pi], # , initVelRange  = [-12,12],
                                  vMax = 8, aMax = 20)
h = 10/200 
t = np.linspace(-1+h, 8, 200)
simModelTest.CreateModel()
fig, axs = plt.subplots(2)


for i in range(3): 
    vecTest = simModelTest.CreateInputVector()
    phiInp = AccumulateAngle(np.arctan2(vecTest[1,:], vecTest[0,:]))
    axs[0].plot(t, phiInp, label='trajectory ' + str(i+1))
    axs[1].plot(t[1:], np.diff(phiInp)/np.diff(simModelTest.timeVecIn), label='traj ' + str(i+1)) # visualize velocity by numerical integration
# plt.title('vel profile')

# solRef = simModelTest.ComputeModel(vecTest) # calculate solution to input vecTest
# axs[0].plot(solRef)
axs[0].set_ylabel(r'$\varphi$ in rad')
axs[1].set_ylabel(r'$\omega$ in rad/s')
axs[1].set_xlabel(r'$t$ in s')
axs[0].legend()
axs[1].plot([-1,8], [-8]*2, 'k--')
axs[1].plot([-1,8], [8]*2, 'k--')
axs[0].grid()
axs[1].grid()
plt.tight_layout()
