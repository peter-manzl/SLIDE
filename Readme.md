## SLIDE

The method SLiding-window Initially-truncated Dynamic-response Estimator (SLIDE) is a deep-learning based method for estimating the output of mechanical and multibody system.   
The corresponding research paper is now available on arxiv, doi: [arXiv.2409.18272](https://doi.org/10.48550/arXiv.2409.18272) and in the submission process for journal publication. 

In the development following versions of packages were used: 
* Python 3.11.8
* pytorch 2.2.1 (+cu121)
* exudyn 1.8.52 (and newer)
* matplotlib 
* numpy 1.23.5  
* ngsolve 6.2.2403

The according `requirements.txt` file can be used to install the requirements with  
`pip install -r requirements.txt`   
For the flexible 6R robot example additionally ngsolve is required. We used Version 6.2.2403. 

### Exudyn
Exudyn is a flexible multibody dynamics simulation code. The C+ core ensures efficient simulation, while the Python interface enables compability with pytorch and other machine learning tools. 
For more information see the [extensive documentation](https://exudyn.readthedocs.io/en/latest/docs/RST/Exudyn.html) and the [examples](https://github.com/jgerstmayr/EXUDYN/tree/master/main/pythonDev/Examples) on github.


### Source files: 
Please note that the provided source code is experimental research. There are both models in the `simModels` and features in the `AISurrogateLib` which are not part of the publication and may not work. 
The object `NeuralNetworkTrainingCenter` is a container object for both simulation and neural networks. It is denoted nntc in the code and enables parameter variations, thus training a number of neural networks in parallel with different parameter settings. 

* `simModels`: here the class definitions for the simulation models and some helper / utility functions are implemented.
    * `SimulationModel`: parent class to derive all other simulation models from. 
    * `Oscillator`: both linear and Duffing oscillator, depending on the provided parameters. 
    * `OscillatorContinuous`: same as oscillator, but truncation is done directly to the data. 
    * `SliderCrank`: Slider-crank system. By setting the flexible flag the connecting rod is modeled with ANCF elements. Also rigid body simulation can be switched on, but then the output has to be changed to outputType=0, which measures the slider's position. 
    * `Flex6RRobot`: The puma 560 standing on a flexible socket. 


* `AISurrogateLib`: Base library providing "presets" for neural networks. 


## examples
Please note that all examples also create the associated data from the simulation. While the oscillators run very fast, creating data for the flexible system may take longer. In the paper, data creation was done partly on a cluster (Leo5), while  the training itself was performed locally. Data is only created if the according file is not existing yet. 

* `01_LinearOscillator.py`: learning of linear oscillator system. 
* `02_DuffingOscillator.py`: learning of Duffing oscillator including the error estimator using the SLIDE method. 
* `03_SliderCrankFlexible.py`: the flexible slider-crank system including error estimator. 
* `04_Flexible_6RRobot.py`: the Flexible Robot including error estimator. 

## Source/data: 


## Source/model: 
In this directory the parameters from the neural networks are saved as *.pth files and can be loaded either with ther utility function of the neural network training center  `LoadNNModel`. 

## Source/solution: 
Here training and validation loss (MSE) is saved for plotting. 

## Licence 
See `Licence.txt`.