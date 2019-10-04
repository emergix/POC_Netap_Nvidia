# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:31:17 2019

@author: Sergey
"""

###########################################################
###########################################################

from __future__ import print_function
import sys, os

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sys import platform
from decimal import *
from sklearn.model_selection import train_test_split

path='./'
if platform == 'win32' :
    path ='/workspace'
    #path ='C:/Users/Sergey/Desktop/Natixis/Yeti'


import keras, tensorflow, pkg_resources
dataDirectory = 'Data'
dataLearningFile = "1dim_VolLoc-Example-new.CSV"
dataNotationFile = "1dim_VolLoc-Example-new.CSV"

os.chdir(path)


import tensorflow as tf
from math import sqrt, exp, log, erf,floor
import numpy 
import pandas as pd
import keras.backend as K
import horovod.keras as hvd

import keras
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from keras import callbacks
from keras.layers import Activation
from keras.layers import Lambda
from keras.models import Sequential
from keras.models import model_from_json
import keras.layers
import matplotlib
import matplotlib.pyplot as pyplot
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import mpl_toolkits
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import preprocessing
from sklearn import model_selection
import joblib
import seaborn as sns
import scipy
from IPython.display import display
import time
#%matplotlib inline
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual,FloatProgress
import ipywidgets as widgets
import math
import threading
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import copy
from mpi4py import MPI
# Generate dummy data



getcontext().prec = 8
from util_functions_YetiPhen_VolLoc import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pkg_resources

ListField1 ={"S1", "mu1", "bonus", "YetiBarrier", "YetiCoupon", "PhoenixBarrier","PhoenixCoupon","PDIBarrier","PDIGearing","PDIStrike","PDIType",
             "maturity","nbDates"}

ListField2={"vol-date0-strike0","vol-date0-strike1","vol-date0-strike2","vol-date0-strike3","vol-date0-strike4","vol-date0-strike5","vol-date0-strike6",
           "vol-date0-strike7","vol-date0-strike8","vol-date0-strike9","vol-date0-strike10","vol-date0-strike11","vol-date0-strike12","vol-date0-strike13",
           "vol-date0-strike14","vol-date1-strike0","vol-date1-strike1","vol-date1-strike2","vol-date1-strike3","vol-date1-strike4","vol-date1-strike5",
           "vol-date1-strike6","vol-date1-strike7","vol-date1-strike8","vol-date1-strike9","vol-date1-strike10","vol-date1-strike11","vol-date1-strike12",
           "vol-date1-strike13","vol-date1-strike14","vol-date2-strike0","vol-date2-strike1","vol-date2-strike2","vol-date2-strike3","vol-date2-strike4",
           "vol-date2-strike5","vol-date2-strike6","vol-date2-strike7","vol-date2-strike8","vol-date2-strike9","vol-date2-strike10","vol-date2-strike11",
           "vol-date2-strike12","vol-date2-strike13","vol-date2-strike14","vol-date3-strike0","vol-date3-strike1","vol-date3-strike2","vol-date3-strike3",
           "vol-date3-strike4","vol-date3-strike5","vol-date3-strike6","vol-date3-strike7","vol-date3-strike8","vol-date3-strike9","vol-date3-strike10",
           "vol-date3-strike11","vol-date3-strike12","vol-date3-strike13","vol-date3-strike14","vol-date4-strike0","vol-date4-strike1","vol-date4-strike2",
           "vol-date4-strike3","vol-date4-strike4","vol-date4-strike5","vol-date4-strike6","vol-date4-strike7","vol-date4-strike8","vol-date4-strike9",
           "vol-date4-strike10","vol-date4-strike11","vol-date4-strike12","vol-date4-strike13","vol-date4-strike14","vol-date5-strike0","vol-date5-strike1",
           "vol-date5-strike2","vol-date5-strike3","vol-date5-strike4","vol-date5-strike5","vol-date5-strike6","vol-date5-strike7","vol-date5-strike8",
           "vol-date5-strike9","vol-date5-strike10","vol-date5-strike11","vol-date5-strike12","vol-date5-strike13","vol-date5-strike14","vol-date6-strike0",
           "vol-date6-strike1","vol-date6-strike2","vol-date6-strike3","vol-date6-strike4","vol-date6-strike5","vol-date6-strike6","vol-date6-strike7",
           "vol-date6-strike8","vol-date6-strike9","vol-date6-strike10","vol-date6-strike11","vol-date6-strike12","vol-date6-strike13","vol-date6-strike14",
           "vol-date7-strike0","vol-date7-strike1","vol-date7-strike2","vol-date7-strike3","vol-date7-strike4","vol-date7-strike5","vol-date7-strike6",
           "vol-date7-strike7","vol-date7-strike8","vol-date7-strike9","vol-date7-strike10","vol-date7-strike11","vol-date7-strike12","vol-date7-strike13",
           "vol-date7-strike14","vol-date8-strike0","vol-date8-strike1","vol-date8-strike2","vol-date8-strike3","vol-date8-strike4","vol-date8-strike5",
           "vol-date8-strike6","vol-date8-strike7","vol-date8-strike8","vol-date8-strike9","vol-date8-strike10","vol-date8-strike11","vol-date8-strike12",
           "vol-date8-strike13","vol-date8-strike14","vol-date9-strike0","vol-date9-strike1","vol-date9-strike2","vol-date9-strike3","vol-date9-strike4",
           "vol-date9-strike5","vol-date9-strike6","vol-date9-strike7","vol-date9-strike8","vol-date9-strike9","vol-date9-strike10","vol-date9-strike11",
           "vol-date9-strike12","vol-date9-strike13","vol-date9-strike14","vol-date10-strike0","vol-date10-strike1","vol-date10-strike2","vol-date10-strike3",
           "vol-date10-strike4","vol-date10-strike5","vol-date10-strike6","vol-date10-strike7","vol-date10-strike8","vol-date10-strike9","vol-date10-strike10",
           "vol-date10-strike11","vol-date10-strike12","vol-date10-strike13","vol-date10-strike14","vol-date11-strike0","vol-date11-strike1","vol-date11-strike2",
           "vol-date11-strike3","vol-date11-strike4","vol-date11-strike5","vol-date11-strike6","vol-date11-strike7",
           "vol-date11-strike8","vol-date11-strike9","vol-date11-strike10","vol-date11-strike11","vol-date11-strike12","vol-date11-strike13","vol-date11-strike14"}

dataDirectory = 'Data'
dataLearningFile = "1dim_VolLoc-Example-new.CSV"
dataNotationFile = "1dim_VolLoc-Example-new.CSV"

params = metaparameters()

params.INPUT_DIM  = 192
params.INPUT_OPTION = 192
params.INPUT_VOL = 13
params.NB_NEURON_PRINCIPAL = 8
params.ACTIVATION_PRINCIPALE = 'tanh'
params.ACTIVATION_PRINCIPALE_FINALE = 'linear'
params.INITIAL_LEARNING_NB_EPOCH = 1000
params.LEARNINGBASE_ORIGIN = "New_Test"
params.LEARNINGBASE_BUT = "New_Simu_Test"
params.GENETIC_LEARNING_NB_EPOCH = 10
params.BATCH_SIZE_PRINCIPAL = 65536#8192#32768#16384#
params.OPTIMIZER = 'adamax'##############################
params.OPTIMIZER_GENETIC = 'SGD'###########################
params.NBLAYERS = 11
params.NB_LOOPS = 5
params.PATH = path
params.VERBOSE_FLAG = 0
params.EPSILON_GREEDINESS = 0.25
params.EPSILON_GREEDINESS_DECREASING_FACTOR = 0.99
params.INITIAL_NETWORK_STRUCTURE = [[0,1000], [0,800], [0,400], [0,200],[0,200], [0,100], [0,100], [0,50],[0,25], [0,10]]
params.NOTATION_FILE = dataDirectory + '/' + dataNotationFile
params.LISTFIELD1 = ListField1
params.LISTFIELD2 = ListField2



params.INITIAL_NETWORK_STRUCTURE = [[0,1000], [0,800], [0,400], [0,200],[0,200], [0,100], [0,100], [0,50],[0,25], [0,10]]

#################################################################################################
#################################################################################################

def read_and_concat(files):
    
    dfs = []
    
    for file in files:
        df = pd.read_csv(file, sep = ';', na_filter = False)
        dfs.append(df)
        
    concat = pd.concat(dfs)
    concat = concat.reset_index(drop = 'index')
    
    #concat = concat.iloc[:,1:]
    Y = concat.price.values.reshape((-1, 1))
    X = concat.drop(columns = ['nbDates', 'price']).values
    
    return X, Y

def read_all_files_scale_and_split(calib_files, train_files, val_files, test_files):
    
    X_calib, Y_calib = read_and_concat(calib_files)
    X_train, Y_train = read_and_concat(train_files)
    X_val, Y_val = read_and_concat(val_files)
    X_test, Y_test = read_and_concat(test_files)
    
    Y_scaler = preprocessing.MinMaxScaler(feature_range = (0, 1))
    Y_scaler.fit(Y_train)
    
    X_scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
    X_scaler.fit(X_train)
    
    X_calib_scaled = X_scaler.transform(X_calib)
    X_train_scaled = X_scaler.transform(X_train)
    X_val_scaled = X_scaler.transform(X_val)
    X_test_scaled = X_scaler.transform(X_test)
    
    Y_calib_scaled = Y_scaler.transform(Y_calib)
    Y_train_scaled = Y_scaler.transform(Y_train)
    Y_val_scaled = Y_scaler.transform(Y_val)
    Y_test_scaled = Y_scaler.transform(Y_test)
    
    Y_test_orig = Y_test
    
    return (X_scaler, Y_scaler), (X_calib_scaled, Y_calib_scaled),\
            (X_train_scaled, Y_train_scaled), (X_val_scaled, Y_val_scaled),\
             (X_test_scaled, Y_test_scaled), Y_test_orig
####################################################################################


#################################################################################
#################################################################################



class DataGenerator_N(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, n_dims = 16, batch_size = params.BATCH_SIZE_PRINCIPAL, mode = 'train'):
        
        'Initialization'
        
        self.batch_size = batch_size
        
        self.on_epoch_end()
               
        self.n_dims = n_dims
        
        assert(mode in {'train', 'val', 'test'})
        self.mode = mode
        
        if mode == 'train':
            self.data_X = params.X_train_scaled
            self.data_Y = np.squeeze(np.repeat(params.Y_train_scaled[:, np.newaxis, :], self.n_dims, axis = 1))
        elif mode == 'val':
            self.data_X = params.X_val_scaled
            self.data_Y = np.squeeze(np.repeat(params.Y_val_scaled[:, np.newaxis, :], self.n_dims, axis = 1))
        else:
            self.data_X = params.X_test_scaled
            self.data_Y = params.Y_test_scaled
            
 

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_X) / self.batch_size))

    def __getitem__(self, index): 
        
        X = self.data_X[index * self.batch_size : (index + 1) * self.batch_size]
        Y = self.data_Y[index * self.batch_size : (index + 1) * self.batch_size]
        
        #if self.mode in {'train', 'val'}:
        #    Y = np.squeeze(np.repeat(Y[:, np.newaxis, :], self.n_dims, axis = 1))
        #else:
        #    Y = [Y for i in range(self.n_dims)]
        if self.mode == 'test':
            Y = [Y] * self.n_dims
            
        return X, Y

###############################################################################
###############################################################################
        


def CountIndividuals(a):
    
    ii = 0
    for f in os.listdir(a):
        ii = ii + 1
        
    return ii

def convergeIndividualModel(model2):
    
    print('convergeIndividualModel :')
    
    return None
   

def pathCreatechild(TimePath, individualpath, child):
    
    pathCreated = TimePath + '/individu' +  str(child)
    
    if params.rank == 0:
        if not(os.path.isdir(pathCreated)):
            os.makedirs(pathCreated, 0o777)
    
    #print('pathCreatechild=', pathCreated)
    
    return pathCreated

def killIndividual(individualPath):
    
    #print("killing :", individualPath)
    shutil.rmtree(individualPath)
    
def saveIndividualModel(model, specList, path, cnfigName):
    
    model.save_weights(path + '/' + cnfigName + '.hdf5')

    model_json = model.to_json()
    
    with open(path + '/' + cnfigName +'.json', 'w') as json_file:
        json_file.write(model_json)

    filename = path + '/' + cnfigName + 'current_structure.str'
    _= joblib.dump(specList, filename, compress = 9)
    
    filename2 = path + '/' + cnfigName  + 'optim_weights.str'
    optim_weights = joblib.dump(model.optimizer.get_weights(), filename2, compress = 9)
    
    #print('saveIndividualModel model2 at :', path)

    
def loadIndividualModel(IndividuPath, cnfigName):
    
    json_file = open(IndividuPath + "/" + cnfigName  + '.json', 'r')
    file = IndividuPath + '/' + cnfigName  + '.json'
    loaded_model_json = json_file.read()
    json_file.close()  
    
    loaded_model = model_from_json(loaded_model_json)  
    
    filename2 = IndividuPath + '/' + cnfigName  + 'current_structure.str'
    specList = joblib.load(filename2)
    
    #print('loadInvidualModel :IndividuPath=', IndividuPath)
    
    filename2 = IndividuPath + '/' + cnfigName  + 'optim_weights.str'
    optim_weights = joblib.load(filename2)
    
    return loaded_model, specList, optim_weights 
    

    
    


def copyModelFromInitial(originalModelPath, IndividuPath):
    
    shutil.copyfile(originalModelPath + 'current_structure.str', IndividuPath + 'current_structure.str')
 
    

def Idealpopulation(nstep):
    return 16 * 2**nstep
#     if (nstep < 4): 
#         return 3 * nstep + 2
#     else:
#         return 20
    


###############################################################################
###############################################################################
    
def buildModel(specificationList) : 

    Nblayer = len(specificationList)
    inData = Input(shape = (params.INPUT_DIM,))
    layer0 = inData
    precedinnbneuron = params.INPUT_DIM
    
    for klayer in range(Nblayer) :

        nbneuron = specificationList[klayer][1]
        
        if (specificationList[klayer][0] == 0):
            
            layer0 = Dense(nbneuron, activation = params.ACTIVATION_PRINCIPALE, name = 'Layer' + str(klayer))(layer0)
            precedinnbneuron = nbneuron        
            
        elif (specificationList[klayer][0] == 1): 
            
            layer1 = Dense(nbneuron, activation = params.ACTIVATION_PRINCIPALE, name = 'ResnetLayer1_type1' + str(klayer))(layer0)
            layer2 = Dense(precedinnbneuron, use_bias = False, name = 'ResnetLayer_Multiplier' + str(klayer))(layer1)

            added = keras.layers.Add()([layer2, layer0])
            layer0 = added
            precedinnbneuron = precedinnbneuron   
            
        elif (specificationList[klayer][0] == 2): 
            
            layer1 = Dense(nbneuron,\
                   activation = params.ACTIVATION_PRINCIPALE, name = 'ResnetLayer1_type2' + str(klayer))(layer0)
            layer2 = Dense(precedinnbneuron,\
                   activation = params.ACTIVATION_PRINCIPALE, name = 'ResnetLayer1_Multiplier' + str(klayer))(layer1)
            layerParralel = Dense(precedinnbneuron, use_bias = False, name = 'ResnetLayer2_Parralel' + str(klayer))(layer0)
            
            added = keras.layers.Add()([layer2, layer0])
            added = keras.layers.Add()([added, layerParralel])
            layer0 = added
            precedinnbneuron = precedinnbneuron       
            
    prixfinal = Dense(1, activation = params.ACTIVATION_PRINCIPALE_FINALE)(layer0)
    model2 = Model(inputs = inData, outputs = prixfinal)
    
    return model2
    
  


def subcopy3(wL1, wL2, pL): 

    w2_vals = K.batch_get_value(wL2)
    
    to_update = []
    for iparam in range(len(pL)):
    
        if(len(pl[i]) == 1):
        
            w2_val = w2_vals[iparams + 1]
            w1_val = wL1[iparams + 1]#K.batch_get_value([w1])[0]
            
            w2_val[0:dims[0]] = w1_val[0:dims[0]]
            w2_val[dims[0] : w2_val.shape[0]] = 0  
            
            to_update.apppend((wL2[iparams + 1], w2_val])   
            
        if(len(dims) == 2):
        
            w2_val = w2_vals[iparams + 1]
            w1_val = wL1[iparams + 1]#K.batch_get_value([w1])[0]  
            
            w2_val[0:dims[0], 0:dims[1]] = w1_val[0:dims[0], 0:dims[1]]
            w2_val[dims[0]:w2_val.shape[0], 0:w2_val.shape[1]] = 0
            w2_val[0:dims[0], dims[1]:w2_val.shape[1]] = 0      
            
            to_update.apppend((wL2[iparams + 1], w2_val])
    
    K.batch_set_value(to_update)

        
def subcopy2(w1, w2, dims):

    if(len(dims) == 1):
        w2[0:dims[0]] = w1[0:dims[0]]
        w2[dims[0]:w2.shape[0]] = 0
    if(len(dims) == 2):
        w2[0:dims[0], 0:dims[1]] = w1[0:dims[0], 0:dims[1]]
        w2[dims[0]:w2.shape[0], 0 : w2.shape[1]] = 0
        w2[0:dims[0], dims[1]:w2.shape[1]] = 0
        
    return w2   
 
def ComputeparamInsertDebut2(param_list, ResnetRank):

    ll = len(param_list[0:ResnetRank])
    
    def compute(j):
        if   (param_list[j][0] == 0):
            return 1
        elif (param_list[j][0] == 1):
            return 3
        elif (param_list[j][0] == 2):
            return 5       
            
    return sum([compute(i) for i in range(ll)])
 
def setModelOptimizerInsertion(model, optimstate, modelspecList, ResnetRank):

    weights1 = model.get_weights()
    param_list1 = list(map(lambda k: k.shape, weights1))
    paramInsertDebut = ComputeparamInsertDebut2(modelspecList, ResnetRank) + 1
    gapRank = ResnetWidth2(ResnetType)
    
    model.compile(loss = 'mse', optimizer = params.OPTIMIZER_GENETIC.create_instance())

    to_update = []
    for i in range(paramInsertDebut):
    
        to_update.append((model.optimizer.weights[i + 1], optimstate[i + 1]))  
        
    keras.backend.batch_set_value(to_update)
    
    to_update = []
    for i in range(paramInsertDebut + gapRank + 1, len(model.optimizer.weights)):
    
        to_update.append((model.optimizer.weights[i], optimstate[i - gapRank]))
        
    keras.backend.batch_set_value(to_update)
    
    return model 
 
def injectionModelWithInsertion2(model, modelspecList, ResnetRank, ResnetWeight0, ResnetType, optimstate) : 
    
    weights1 = model.get_weights()
    param_list1 = list(map(lambda k: k.shape, weights1))
    paramInsertDebut = ComputeparamInsertDebut2(modelspecList, ResnetRank)
    intermediateDim = param_list1[paramInsertDebut][0]
    
    ResnetWeight = ResnetWeight0 
    modelspeclist2 = copy.deepcopy(modelspecList)
    modelspeclist2.insert(ResnetRank, [ResnetType, ResnetWeight])


    model2 = buidModel2(modelspeclist2, params) 

    model2.compile(loss = 'mse', optimizer = params.OPTIMIZER_GENETIC.create_instance())       
    
    weights2 = model2.get_weights()
    param_list2 = list(map(lambda k: k.shape, weights2))

    paramInsertDebut = ComputeparamInsertDebut2(modelspecList, ResnetRank) + 1 
    gapRank = ResnetWidth2(ResnetType)
    paramInsertFin = paramInsertDebut + gapRank

    firstset = [param_list1[i] for i in range(paramInsertDebut)]
    lastset = [param_list1[i] for i in range(paramInsertDebut, len(param_list1))]

    for iparam in range(paramInsertDebut):    
        w = subcopy2(weights1[iparam], weights2[iparam], param_list1[iparam])
        weights2[iparam] = w
        
    for j in range(gapRank)  :      
    
        if (len(param_list2[paramInsertDebut + j]) == 2):
            weights2[paramInsertDebut + j][0:param_list2[paramInsertDebut + j][0], 0:param_list2[paramInsertDebut + j][1]] = 0
        else :
            weights2[paramInsertDebut + j][0:param_list2[paramInsertDebut + j][0]] = 0
            
    for iparam in range(paramInsertDebut, len(param_list1)):
    
        w = subcopy2(weights1[iparam], weights2[iparam + gapRank], param_list1[iparam])
        weights2[iparam + gapRank] = w
        
    model2 = setModelOptimizerInsertion(model2, optimstate, modelspecList, ResnetRank)
    model2.set_weights(weights2) 
    
    return model2, modelspeclist2

def setModelOptimizerDimIncrease(model2, model, param_list, param_list2, optimstate):#  
      
    model2.compile(loss='mse', optimizer = params.OPTIMIZER_GENETIC.create_instance())
    
    #for iparam in range(len(param_list2)):
    subcopy3(optimstate, model2.optimizer.weights, param_list) 
  
    return model2

'''
def setModelOptimizerMinimal(model):      
    model.compile(loss='mse', optimizer = params.OPTIMIZER_GENETIC.create_instance())
    return model
''' 
 
def injectionModel(model, modelspecList, deltaspecList, optimstate) : 

    modelspeclist2 = [[modelspecList[i][0], modelspecList[i][1] + deltaspecList[i]] for i in range(len(modelspecList))]
    model2 = buildModel(modelspeclist2)
 
    #model2 = setModelOptimizerMinimal(model2)
    
    weights1 = model.get_weights()
    param_list = list(map(lambda k: k.shape, weights1))
    weights2 = model2.get_weights()
    param_list2 = list(map(lambda k: k.shape, weights2))
    
    for iparam in range(len(param_list2)):
        w = subcopy2(weights1[iparam], weights2[iparam], param_list[iparam])
        weights2[iparam] = w
        
    model2 = setModelOptimizerDimIncrease(model2, model, param_list, param_list2, optimstate)
    model2.set_weights(weights2)  
    
    return model2, modelspeclist2

def BulleListPar(i, NbLayers, hf):
    return [floor(exp(2.5 - 2 * abs(x - i) / (0.5 * NbLayers))) + 1 for x in range(NbLayers)]



def mutateIndividualModel(model, specList, optimstate):

    ptyp0 = params.RESNETYPE1_MUTATIONPROBABILITY 
    ptyp1 = params.RESNETYPE1_MUTATIONPROBABILITY 
    ptyp2 = params.RESNETYPE2_MUTATIONPROBABILITY 
    hf = params.HEATFACTOR_MUTATION
    
    NbLayers = len(specList)
    deltaNbNeuronList = BulleListPar(0, NbLayers, hf)
   
    model2, speclist2 = injectionModel(model, specList, deltaNbNeuronList, optimstate)#  

    if (random.uniform(0,1) < ptyp1):
        ResnetRank = random.randint(1, NbLayers - 1)
        ResnetWeight = params.MUTATION_RESNETSIZE
        ResnetType = 1
        model4, speclist4 = injectionModelWithInsertion2(model2, speclist2, ResnetRank, ResnetWeight, ResnetType, optimstate)
    else:
        model3, speclist3 = model2, speclist2
        
        if (random.uniform(0,1) < ptyp2):
            ResnetRank = random.randint(1, NbLayers - 1)
            ResnetWeight = params.MUTATION_RESNETSIZE
            ResnetType = 2
            
            model4, speclist4 = injectionModelWithInsertion2(model3, speclist3, ResnetRank, ResnetWeight, ResnetType, optimstate)
        else:
            model4, speclist4 = model3, speclist3
               
    return model4, speclist4

#############################################################################
#############################################################################    

from keras.optimizers import SGD, Adam
import joblib
import pickle
import numpy as np
import gc

class Optimizer:
    
    def __init__(self, opt_type, params):
        
        assert type(params) is dict, "params must be a dict" 
        assert opt_type in {'SGD', 'Adam'}, "this type of optimizer is not supported" 
        
        self.type = opt_type
        
        if self.type == 'SGD':
            for param in params:
                assert param in {'lr', 'momentum', 'decay', 'nesterov'}, 'Illegal name of parameter'
        elif self.type == 'Adam':
            for param in params:
                assert param in {'lr', 'beta_1', 'beta_2', 'epsilon', 'decay', 'amsgrad'}, 'Illegal name of parameter' 
        
        self.params = params
    
    @classmethod   
    def from_file(self, address):
        
        load = joblib.load(address)
        
        return load

    def create_instance(self):
        
        if self.type == 'SGD':
            return SGD(**self.params)
        elif self.type == 'Adam':
            return Adam(**self.params)
        
    def save_to_individu(self, address):
        
        try:
            joblib.dump(self, address)
        except Exception as e:
            print(e)
            
    def introduce_mutation(self):
        
        if self.type == 'SGD':
            pass
        elif self.type == 'Adam':
            pass
        
        r = np.random.rand()
        
        if r < 0.15:
            self.params['lr'] *= 2.0
        elif r < 0.55:
            self.params['lr'] *= 1.0
        else:
            self.params['lr'] *= 0.5
            
            
class Task:
    def __init__(self, ModelPath, individuPath, cnfigName):
        
        self.ModelPath = ModelPath 
        self.IndividuPath = individuPath
        self.cnfigName = cnfigName 
                
###############################################################################
###############################################################################
        
class Best_Model:
    def __init__(self, task):
        
        model, specList, opt_weights = loadIndividualModel(task.ModelPath, task.cnfigName)
        model2, specList2 = mutateIndividualModel(model, specList, opt_weights)
        
        self.optimizer_weights = model2.optimizer.get_weights()
        self.optim_lengths = (len(self.optimizer_weights) - 1) // 3
        
        self.parent_model = model
        self.parent_speclist = specList
        self.parent_weights = model.get_weights()
        
        self.model = model2
        self.speclist = specList2
        self.init_weights = model2.get_weights()
        
        with open(task.ModelPath + '/Best_score.txt', 'r+') as fp:
            parent_score = float(fp.read())
            
        self.parent_score = parent_score
        
        self.task = task

        #_________________updated values!_________________________________________
        self.best_weights = model2.get_weights()
        
        # important!
        self.best_score = parent_score

        self.best_optimizer = None
        self.epoch = 'Not Updated'
        
class CheckScoreSubmodels(tf.keras.callbacks.Callback):


    def __init__(self, evaluate_model, best_weights, optimizer):
        assert(type(best_weights[0]) is Best_Model)
        
        self.best_weights = best_weights
        self.e_model = evaluate_model
        
        self.test_generator = DataGenerator_N(n_dims = len(best_weights), mode = 'test')
        self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs = None):
        
        
        scores = self.e_model.evaluate_generator(self.test_generator)
        #print('EVALUATING SUBMODELS...', scores)
        
        assert(len(scores[1:]) == len(self.best_weights))
        assert(np.allclose(scores[0], np.sum(scores[1:]), atol = 0.01))
        
        
        for i in range(len(self.best_weights)):
            
            if self.best_weights[i].best_score > scores[i]:
                
                self.best_weights[i].best_weights = self.best_weights[i].model.get_weights()
                self.best_weights[i].best_optimizer =  self.optimizer
                self.best_weights[i].epoch = epoch
                self.best_weights[i].best_score = scores[i]
    
###############################################################################
###############################################################################
    
def recopy_weights_params(metamodel, weights_individuals_list):
    
    if isinstance(metamodel.optimizer, keras.optimizers.Adam):
        
        weights = getattr(metamodel.optimizer,'weights')
        shapes_meta = list(map(lambda x:x.shape, weights[1:]))
        
        lengths = [(len(x) - 1) // 3 for x in weights_individuals_list]
        count = 0
        
        means = [weights_individuals_list[i][1:lengths[i] + 1]\
                 for i in range(len(weights_individuals_list))]
        
        varrs = [weights_individuals_list[i][lengths[i] + 1 : 2 * lengths[i] + 1] for\
                 i in range(len(weights_individuals_list))]
        
        caps = [weights_individuals_list[i][2 * lengths[i] + 1 : 3 * lengths[i] + 1] for\
                i in range(len(weights_individuals_list))]
        
        weights_individuals = sum( means + varrs + caps, [])
        #print(weights_individuals[0])
        shapes_ind = list(map(lambda x:x.shape, weights_individuals))
        
        assert len(shapes_ind) == len(shapes_meta)
        
        for x, y in zip(shapes_meta, shapes_ind):
            if len(x) == 2:
                assert x[0] == y[0]
                assert x[1] == y[1]
            else:
                assert x[0] == y[0]
        
        to_update = []
          
        for x,y in zip(weights[1:], weights_individuals):
            to_update.append((x,y))

        K.batch_set_value(to_update) 
        
        
    elif isinstance(metamodel.optimizer, keras.optimizers.SGD):
        
        weights = getattr(metamodel.optimizer,'weights')
        shapes_meta = list(map(lambda x:x.shape, weights[1:]))
        
        weights_individuals = sum([weights_individuals_list[i][1:] for\
                                       i in range(len(weights_individuals_list))],[])
          
        shapes_ind = list(map(lambda x:x.shape, weights_individuals))
        
        assert len(shapes_ind) == len(shapes_meta)
        
        for x, y in zip(shapes_meta, shapes_ind):
            if len(x) == 2:
                assert x[0] == y[0]
                assert x[1] == y[1]
            else:
                assert x[0] == y[0]
        
        to_update = []
        
        for x,y in zip(weights[1:], weights_individuals):
            to_update.append((x,y))

        K.batch_set_value(to_update)   
        

    else:
        print('The optimizer type is not appropriate')

def extract_optimizer_state(lengths, metamodel):
    
    MODELS_PARAMS = []
    
    def split_t(lgths, array):
        splt = []
        prev = 0
        
        for i in range(len(lgths)):
            splt.append(array[prev:lgths[i]])
            prev = lgths[i]
        
        
        #splt.append(array[lengths[-1]:])

        return splt
    
    if isinstance(metamodel.optimizer, keras.optimizers.Adam):
        
        models_opt_means = []
        models_opt_vars = []
        models_opt_caps = []

        cs = list(np.cumsum(lengths))
        
        SECTION_1 = cs[-1] + 1
        SECTION_2 = cs[-1] * 2 + 1
        
        METAPARAMS = metamodel.optimizer.get_weights()
        
        
        for i, x in enumerate(split_t(cs, METAPARAMS[1:SECTION_1])):
            models_opt_means.append(x)
        print(len(models_opt_means))
        for i, x in enumerate(split_t(cs, METAPARAMS[SECTION_1:SECTION_2])):
            models_opt_vars.append(x)
        
        for i, x in enumerate(split_t(cs, METAPARAMS[SECTION_2:])):
            models_opt_caps.append(x)
            
        MODELS_PARAMS = []

        
        for i in range(len(lengths)):
            MODELS_PARAMS.append([0] + models_opt_means[i] + models_opt_vars[i] + models_opt_caps[i])
        
    elif isinstance(metamodel.optimizer, keras.optimizers.SGD):
        pass
    else:
        print('inappropriate optimizer')
    
    return MODELS_PARAMS


def metamodels_create_and_train(tasks, list_optimizers, limit_tasks_by_model = 16):
    
    #print('N_TASKS: ', len(tasks))
    batch_tasks = [tasks[i * limit_tasks_by_model : (i + 1) * limit_tasks_by_model]\
                   for i in range(len(tasks) // limit_tasks_by_model + 1)]
    
    TOTAL_BEST = {}
    
    for batch_task in batch_tasks:
        if len(batch_task) == 0:
            break
            
        print('Loading data from parents and creating individuals....') 
        
        best_models = [Best_Model(task) for task in batch_task]
        
        for opt in list_optimizers:
            
            optimizer = opt.create_instance()#hvd.DistributedOptimizer(opt.create_instance())
            
            list_models = [bm.model for bm in best_models]
            
            print('Creating metamodel...')
            metamodel, eval_model = create_metamodel(list_models, optimizer)
            
            print( 'Copying optimizers weights')
            lists_params_optimizers = [bm.optimizer_weights for bm in best_models]
            
            recopy_weights_params(metamodel, lists_params_optimizers)
            
            cbs = []
            #if hvd.rank() == 0:
            
            #es = keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 1500, verbose=0,\
            #                                      mode ='min', baseline=None, restore_best_weights=False)
            
            cbs.append(CheckScoreSubmodels(eval_model, best_models, opt))
            
            print('Generator creation...')
            train_generator = DataGenerator_N(n_dims = len(list_models), mode = 'train')
            val_generator = DataGenerator_N(n_dims = len(list_models), mode = 'val')
            
            fit_params_gen = {'verbose':0, 'epochs':1, 'shuffle':False, 'use_multiprocessing':True, 'workers':16}
            
            #fit_params = {'verbose':1, 'epochs':1, 'shuffle': True}
            
            #nbatches_train, mod = divmod(len(params.X_train_scaled), params.BATCH_SIZE_PRINCIPAL)
            #nbatches_valid, mod = divmod(len(params.X_val_scaled), params.BATCH_SIZE_PRINCIPAL)
            
            #print('####################### FITTING METAMODEL FOR %s#############################'%opt.type)
            
            
            print('Fitting metamodel...')
            history = metamodel.fit_generator(generator = train_generator, \
                                              validation_data = val_generator, \
                                       callbacks = cbs, **fit_params_gen)
            
            #  steps_per_epoch = nbatches_train // hvd.size(), validation_steps = 3 * nbatches_valid // hvd.size(),
            #history = metamodel.fit(params.X_train_scaled, params.Y_train_scaled,\
            #                        validation_data = (params.X_val_scaled, params.Y_val_scaled),\
            #                           callbacks = callbacks, **fit_params)
            #print(metamodel.summary())
            for bm in best_models:
                bm.model.set_weights(bm.init_weights)
        
        print('Extracting optimizers states...')
        
        lgths_optimizer_pars = [bm.optim_lengths for bm in best_models]
        extracted_optimizers_states = extract_optimizer_state(lgths_optimizer_pars, metamodel)
        
        print('Saving all to the disk...')
        for idxx, bm in enumerate(best_models):
            
            if not bm.best_optimizer:
            
                bm.model.optimizer.set_weights(extracted_optimizers_states[idxx])
                bm.parent_model.set_weights(bm.parent_weights)
                
                with open(bm.task.IndividuPath + '/Report.txt', 'w+') as fp:
                    fp.write('Not Evolved from ' + bm.task.ModelPath + '\n Parent speclist: ' + str(bm.parent_speclist))
                
                with open(bm.task.IndividuPath + '/Best_score.txt', 'w+') as fp:
                    fp.write(str(bm.parent_score))
                    
                saveIndividualModel(bm.parent_model, bm.parent_speclist, bm.task.IndividuPath, bm.task.cnfigName)
                
                TOTAL_BEST[bm.task.IndividuPath] = bm.best_score
                
            else:
                msg = 'Evolved from '+ bm.task.ModelPath + ': ' + str(bm.parent_score) + \
                                    '-------> ' + str(bm.best_score) + '\n'
                msg = msg + 'Mutation: ' + str(bm.parent_speclist) + '-------> ' + str(bm.speclist) + '\n'
                
                msg = msg + 'Best optimizer: ' + bm.best_optimizer.type + ' ' + str(bm.best_optimizer.params) + '\n'
                
                msg = msg + 'Best epoch: ' + str(bm.epoch)
                
                with open(bm.task.IndividuPath + '/Report.txt', 'w+') as fp:
                    fp.write(msg)
                
                with open(bm.task.IndividuPath + '/Best_score.txt', 'w+') as fp:
                    fp.write(str(bm.best_score))
                    
                bm.model.set_weights(bm.best_weights)
                bm.model.optimizer.set_weights(extracted_optimizers_states[idxx])
                saveIndividualModel(bm.model, bm.speclist, bm.task.IndividuPath, bm.task.cnfigName)
                
                TOTAL_BEST[bm.task.IndividuPath] = bm.best_score
        
    return TOTAL_BEST

def create_metamodel(list_models, optimizer):
    
    #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', list_models)
    inp = Input(shape = (192,))
    
    outs_list = [model(inp) for model in list_models]
    
    concat = Lambda(lambda l: K.concatenate(l, axis = -1))
    
    outs = concat(outs_list)
    
    assert(outs.shape[-1] == len(list_models))
    assert(K.ndim(outs) == 2)
    
    
    metaModel = Model(inputs = inp, outputs = outs)
    
    evaluate_model = Model(inputs = inp, outputs = outs_list)
    
    namespace = evaluate_model.output_names
    #print(evaluate_model.output_names)
    
    #G = 8
    #metaModel = keras.utils.multi_gpu_model(metaModel, gpus = G)
    metaModel.compile(loss = 'mse', optimizer = optimizer)
    #print(metaModel.summary())
    evaluate_model.compile(loss = {namespace[i] : 'mse' for i in range(len(list_models))}, optimizer = optimizer)
    
    return metaModel, evaluate_model

###############################################################################
###############################################################################

def ReinforceOptimalityWithGenetic(list_optimizers, pkgNameOriginal, pkgNameBut, InitialNbIndividual, populationctrlFunc, nbChildAllowed,\
                         nbloop, cnfigName, params, save = True, generate_errors = False):
    
    originalModelPath = params.PATH + "/" + pkgNameOriginal   


    SimuPath = params.PATH + "/" + pkgNameBut 
    if params.rank == 0:
        if not(os.path.isdir(SimuPath)):
            os.makedirs(SimuPath, 0o777)
    

    SimuPath = SimuPath + '/loop'
    
    if params.rank == 0:
        if (os.path.isdir(SimuPath)):
            shutil.rmtree(SimuPath)

        os.makedirs(SimuPath, 0o777 ) 
    
    precedingIndividualPathList = []    
    originalModelPath = params.PATH + "/" + pkgNameOriginal 
    SimuPath = params.PATH + "/" + pkgNameBut + '/loop'
    
    params.comm.Barrier()
    
    for iTimeStep in range(nbloop):
        start = time.time()
        #print(iTimeStep, 'Tiiiiiiimeeeeee')
        TimePath = SimuPath + "/" + "time" + str(iTimeStep) 
        
        if params.rank == 0:
            if not(os.path.isdir(TimePath)):
                    os.makedirs(TimePath, 0o777)
                    #print("created : ",TimePath)

        LIST_OF_TASKS = []        
        
        params.comm.Barrier()
        
        print('Creating directories and tasks...')
        
        if (iTimeStep == 0) :
            for individual in range(InitialNbIndividual):
                IndividuPath = TimePath +  "/individu" + str(individual)
                
                
                if params.rank == 0:
                    if not(os.path.isdir(IndividuPath)):
                        os.makedirs(IndividuPath, 0o777)
                        #print("created : ",IndividuPath)
                
                
                params.comm.Barrier()
                
                task = Task(originalModelPath, IndividuPath,  cnfigName)

                LIST_OF_TASKS.append(task)

                #precedingIndividualPathList.append(IndividuPath)                     
        else :

            invidualpathList = []
            individuNumero = 0

            for individualpath in precedingIndividualPathList:                            

                for child in range(nbChildAllowed): 

                    childPath = pathCreatechild(TimePath, individualpath, individuNumero)
                    individuNumero += 1
                    
                    params.comm.Barrier()
                    
                    task = Task(individualpath, childPath, cnfigName)

                    LIST_OF_TASKS.append(task)

                    #invidualpathList.append(childPath)
            #precedingIndividualPathList = invidualpathList

            #K.clear_session()
        #if hvd.rank() > 0:
        #    LIST_OF_TASKS = []
            
        #hvd.broadcast_global_variables(0)
        
        
        LIST_OF_TASKS = [LIST_OF_TASKS[i] for i in range(len(LIST_OF_TASKS)) if i % params.size == params.rank]
        
        NOTES_INDIVIDUS = metamodels_create_and_train(LIST_OF_TASKS, list_optimizers,\
                                limit_tasks_by_model = 4)
        
        params.comm.Barrier()
        K.clear_session()
        
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.compat.v1.Session(config = config))
        params.comm.Barrier()
        
        if params.rank == 0:
            
            for fr in range(1, params.size):
                notes = params.comm.recv(source = fr, tag = 10)
                for x in notes:
                    NOTES_INDIVIDUS[x] = notes[x]
        else:
            params.comm.send(NOTES_INDIVIDUS, dest = 0, tag = 10)
        
        
        #K.clear_session()
        
        if params.rank == 0:
            
            print('Selecting best individuals...')
            SORTED_INDIVIDUS = sorted(NOTES_INDIVIDUS, key = NOTES_INDIVIDUS.__getitem__)

            N_survive = populationctrlFunc(iTimeStep)
            bestNoteIndividus = SORTED_INDIVIDUS[:N_survive]
            worstNoteIndividus = SORTED_INDIVIDUS[N_survive:]

            #print('ListOnote=', SORTED_INDIVIDUS)
            #print('bestNoteIndexlist=', bestNoteIndividus)
            #print('worstNoteIndexlist=', worstNoteIndividus)

            for individu in worstNoteIndividus : 
                killIndividual(individu)
            
            
            precedingIndividualPathList = [individu for individu in bestNoteIndividus]
        
        precedingIndividualPathList = params.comm.bcast(precedingIndividualPathList, root = 0)

        end = time.time()
        print("___________________Step %d lasted %f"%(iTimeStep, end - start))
###############################################################################
###############################################################################


try:
    
    path_to = '/mnt/natixis_1/1/'#'/workspace/FirstAttempt/Data/Sergey/'#
    params.list_gen = [path_to + x for x in os.listdir(path_to)]

    # params.calib_files = [params.list_gen[0]]
    # params.train_files = params.list_gen[1:3]
    # params.val_files = params.list_gen[3:4]
    # params.test_files = [params.list_gen[4]]

    params.calib_files = [params.list_gen[0]]
    params.train_files = params.list_gen[1:3]
    params.val_files = params.list_gen[31:33]
    params.test_files = [params.list_gen[33]]

    ####################################################################################
    print('READING DATA...')

    (params.X_scaler, params.Y_scaler), (params.X_calib_scaled, params.Y_calib_scaled\
    ),(params.X_train_scaled, params.Y_train_scaled), (params.X_val_scaled, params.Y_val_scaled\
    ), (params.X_test_scaled, params.Y_test_scaled), params.Y_test_orig = read_all_files_scale_and_split(\
                params.calib_files, params.train_files, params.val_files, params.test_files)

    print(params.X_train_scaled.shape, params.X_calib_scaled.shape, params.X_val_scaled.shape, params.X_test_scaled.shape)
    print('READ DATA')
    ####################################################################################
    ####################################################################################
    print('INIT HOROVOD')
    
    hvd.init()

    params.comm = MPI.COMM_WORLD

    params.size = params.comm.Get_size()
    params.rank = params.comm.Get_rank()

    assert(params.size == hvd.size())
    assert(params.rank == hvd.rank())


    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.compat.v1.Session(config = config))

    nbloop = 5

    nbChildAllowed = 4
    pkgNameOriginal = params.LEARNINGBASE_ORIGIN
    pkgNameBut = params.LEARNINGBASE_BUT
    InitialNbIndividual = 32

    print('INIT_OPTIMIZERS...')
    list_opt = [('Adam', {'lr':1e-7})]#, ('SGD', {'lr':1e-7})
    list_optimizers = [Optimizer(lopt[0], lopt[1]) for lopt in list_opt]

    a = ReinforceOptimalityWithGenetic(list_optimizers, pkgNameOriginal, pkgNameBut,\
                        InitialNbIndividual, Idealpopulation, nbChildAllowed, nbloop, "Vol", params)   
    
except KeyboardInterrupt:
    print('Keeeeyboaaaaard')
    del params
    os._exit()