## SLIDE

The method SLiding-window Initially-truncated Dynamic-response Estimator (SLIDE) is a deep-learning based method for estimating the output of mechanical and multibody system.   
The corresponding research paper is now available on arxiv, doi: [arXiv.2409.18272](https://doi.org/10.48550/arXiv.2409.18272) and in the submission process for journal publication. 

While no specific version for any of the source code is requested, in the development following versions of packages were used: 
* Python 3.11.8
* pytorch 2.2.1 (+cu121)
* exudyn 1.8.52 (and newer)
* matplotlib 
* numpy 1.23.5

For the flexible 6R robot example additionally ngsolve is required. We used Version 6.2.2403. 

### Exudyn
Exudyn is a flexible multibody dynamics simulation code. The C+ core ensures efficient simulation, while the Python interface enables compability with pytorch and other machine learning tools. 
For more information see the [extensive documentation](https://exudyn.readthedocs.io/en/latest/docs/RST/Exudyn.html) and the [examples](https://github.com/jgerstmayr/EXUDYN/tree/master/main/pythonDev/Examples) on github.


### Source files: 
Please note that the provided source code is experimental research. 
The object "NeuralNetworkTrainingCenter" is a container object for both simulation and neural networks. It is denoted nntc in the code and enables parameter variations, thus training a number of neural networks in parallel with different parameter settings. 

* simModels: here the class definitions for the simulation models and some helper / utility functions are implemented.
    * SimulationModel: parent class to derive all other simulation models from. 
    * Oscillator: both linear and Duffing oscillator, depending on the provided parameters. 
    * OscillatorContinuous: same as oscillator, but truncation is done directly to the data. 
    * SliderCrank: 
    * Flex6RRobot: 


* AISurrogateLib: Base library providing "presets" for neural networks. There may be still some "RNN", "CNN" and residual layer implementations which are (currently) not shown in the SLIDE-paper.


## examples
* 01_LinearOscillator
* 02_DuffingOscillator
* 03_SliderCrankFlexible
* 04_Flexible_6RRobot.py

