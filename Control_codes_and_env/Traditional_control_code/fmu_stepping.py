# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:07:23 2019

@author: 20277
"""
import numpy as np
import fmpy
from fmpy import read_model_description
from fmpy.fmi2 import FMU2Slave
from fmpy.simulation import apply_start_values, Recorder


class fmu_stepping:
  """
  This class is used to simulate a FMu file step by step, so input values can be changed during the
  simulation based on output values from the FMU.
  """
  def __init__(self, filename,start_time=None,stop_time=None,sample_time=None,parameters={},input={},output=None):
    """
    Constructor that initializes the FMU and prepares it for simulation.
    
    :param str filename:      filename of the FMU
    :param float start_time:  simulation start time (None: use default experiment or 0 if not defined)
    :param float stop_time:   simulation stop time (None: use default experiment or start_time + 1 if not defined)
    :param float sample_time: how long the simulation shoud run on each step
    :param dict parameters:   constant parameter values. Dictionary of variable name -> value pairs
    :param list input:        list of inputs to be set at each sample time
    :param list output:       list of variables to output each step and to record (None: record model outputs)           
    """
    self.filename = filename
    self.start_time = 0.0 if start_time is None else start_time
    self.stop_time = stop_time
    self.sample_time = sample_time
    self.parameters = parameters
    self.input = input
    self.output = output
    self.input_ref = dict()
    self.input_type = dict()
    self.time = self.start_time
    self.__initialize()


  def step(self, input_values):
    """
    Run the FMU for one step and return the resulting output
    
    :param dict input_values: Dictionary of input name -> value pairs
    
    :return numpy array:      numpy array of the resulting output values after the step
    """
    if self.time == self.start_time:
      self.__startup()
      
    self.__apply_inputs(input_values)
    
    self.fmu.doStep(currentCommunicationPoint=self.time,
                    communicationStepSize=self.sample_time)
    self.time += self.sample_time
    self.recorder.sample(self.time)
    return np.array(self.recorder.rows[-1], dtype=np.dtype(self.recorder.cols))


  def cleanup(self):
    """
    Terminate the FMU and return the record of the output values.
    
    :return numpy array: Simulation result
    """
    self.fmu.terminate()
    self.fmu.freeInstance()
    return self.recorder.result()


  def __initialize(self):
    """
    Load and prepare the FMU.
    Find reference IDs for the input values       
    """
    # read the model description
    self.model_description = read_model_description(self.filename)
    # find stop time
    experiment = self.model_description.defaultExperiment
    if self.stop_time is None:
        if experiment is not None and experiment.stopTime is not None:
            self.stop_time = experiment.stopTime
        else:
            self.stop_time = self.start_time + 1.0

    if self.sample_time is None:
      self.sample_time = (self.stop_time - self.start_time)/1000
    # collect the value references for the input values
    vrs = {}
    vrs_type = {}
    for variable in self.model_description.modelVariables:
      vrs[variable.name] = variable.valueReference
      vrs_type[variable.name] = variable.type
    for key in self.input:
      self.input_ref[key] = vrs.get(key)
      self.input_type[key] = vrs_type.get(key)
    # extract the FMU
    unzipdir = fmpy.extract(self.filename)
    fmu_args = {'guid': self.model_description.guid,
                     'instanceName': None,
                     'unzipDirectory': unzipdir,
                     'modelIdentifier': self.model_description.coSimulation.modelIdentifier}
    # create FMU reference
    self.fmu = FMU2Slave(**fmu_args)


  def __startup(self):
    """
    Initialize the FMU
    Set paramepters
    """
    # setup the FMU
    self.fmu.instantiate()
    self.fmu.setupExperiment(tolerance=None, startTime=self.start_time)
    # set the parametersvalues
    apply_start_values(self.fmu,
                       self.model_description,
                       self.parameters,
                       apply_default_start_values=False)
    # Initialize the FMU
    self.fmu.enterInitializationMode()
    self.fmu.exitInitializationMode()
    # Prepare the record of the output values
    self.recorder = Recorder(fmu=self.fmu,
                        modelDescription=self.model_description,
                        variableNames=self.output,
                        interval=self.sample_time)
    self.recorder.sample(self.time)


  def __apply_inputs(self,input):
    for key in self.input:
      if self.input_type[key] == 'Real':
        self.fmu.setReal([self.input_ref[key]], [float(input[key])])
      elif self.input_type[key] in ['Integer', 'Enumeration']:
        self.fmu.setInteger([self.input_ref[key]], [int(input[key])])
      elif self.input_type[key] == 'Boolean':
        if isinstance(input[key], str):
          if input[key].lower() not in ['true', 'false']:
            raise Exception('The start value "%s" for variable "%s" could not be converted to Boolean' %
                            (input[key], key))
          else:
            input[key] = input[key].lower() == 'true'
        self.fmu.setBoolean([self.input_ref[key]], [bool(input[key])])
      elif self.input_type[key] == 'String':
        self.fmu.setString([self.input_ref[key]], [input[key]])