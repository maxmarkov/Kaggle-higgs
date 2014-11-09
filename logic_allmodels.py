import csv
import sys
import numpy as np
import scipy as sp
import pandas as pd

import sklearn.cross_validation as cv
import os
import inspect
# add path of xgboost python module
code_path = os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0], "../../../python")
sys.path.append(code_path)

import xgboost as xgb

#names = ["KNN", "RF", "GBC","RBFSVM1","LDA",'xboost']
names = ["KNN", "RF", "GBC","LSVM","RBFSVM1","LDA",'xboost']
def read_train_file(mode='train'):
    n_train=250000    
    y_pred_prob=np.zeros((n_train,len(names)))
    y_pred=np.zeros((n_train,len(names)))
    for i in range(len(names)):
     fname='./dataout/'+names[i]+'_'+mode+'.csv'
     data = list(csv.reader(open(fname, "rb"), delimiter=','))
#     X       = np.array([map(float, row[1:-2]) for row in data[1:]])
     idx     = np.array([int(row[0]) for row in data[1:]])
     y_pred_prob[:,i]=np.array([float(row[1]) for row in data[1:]])
     y_pred[:,i]=np.array([float(row[2]) for row in data[1:]])
     y  = np.array([int(row[-1]) for row in data[1:]])
     weights = np.array([float(row[-2]) for row in data[1:]]) 
     
    return y_pred_prob,y_pred,y, weights,idx
 

def read_test_file(mode='test'):
    n_train=550000    
    y_pred_prob=np.zeros((n_train,len(names)))
    y_pred=np.zeros((n_train,len(names)))
    for i in range(len(names)):
     fname='./dataout/'+names[i]+'_'+mode+'_pre.csv'
     data = list(csv.reader(open(fname, "rb"), delimiter=','))
     idx     = np.array([int(row[0]) for row in data[1:]])
     y_pred_prob[:,i]=np.array([float(row[1]) for row in data[1:]])
     y_pred[:,i]=np.array([float(row[2]) for row in data[1:]])     
    return y_pred_prob,y_pred,idx


def save_results_final(idx,reverse_rorder,y_pred,fname):    
    # Write the result list data to a csv file
    y_str=y_bin2str(y_pred)
    df = pd.DataFrame({"EventId": idx, "RankOrder": reverse_rorder,"Class":y_str})
    df.to_csv(fname, index=False, cols=["EventId", "RankOrder", "Class"])


def AMS(s, b):
    '''
    Approximate median significance:
        s = true positive rate
        b = false positive rate
    '''
    assert s >= 0
    assert b >= 0
    bReg = 10.
    return np.sqrt(2.0 * ((s + b + bReg) * np.log(1 + s / (b + bReg)) - s))


def get_rates(prediction, solution, weights):
    '''
    Returns the true and false positive rates.
    This assumes that:
        label 's' corresponds to 1 (int)
        label 'b' corresponds to 0 (int)
    '''
    assert prediction.size == solution.size
    assert prediction.size == weights.size

    # Compute sum of weights for true and false positives
    truePos  = sum(weights[(solution == 1) * (prediction == 1)])
    falsePos = sum(weights[(solution == 0) * (prediction == 1)])

    return truePos, falsePos

def iroot(k, n):
    hi = 1
    while pow(hi, k) < n:
        hi *= 2
    lo = hi / 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        midToK = pow(mid, k)
        if midToK < n:
            lo = mid
        elif n < midToK:
            hi = mid
        else:
            return mid
    if pow(hi, k) == n:
        return hi
    else:
        return lo

def get_training_data(training_file):
    '''
    Loads training data.
    '''
    data = list(csv.reader(open(training_file, "rb"), delimiter=','))
    X       = np.array([map(float, row[1:-2]) for row in data[1:]])
    labels  = np.array([int(row[-1] == 's') for row in data[1:]])
    weights = np.array([float(row[-2]) for row in data[1:]])
    return X, labels, weights
    
def y_bin2str(y_bin):
#  create 's' and 'b' columns from binary input  
   y_str = np.empty(len(y_bin), dtype=np.object)
   y_str[y_bin==1] = 's'
   y_str[y_bin==0] = 'b'
#    y_bin_list=list(y_bin)
#    y_str=[]
#    for x in range(len(y_bin)):
#     y_str.append(['s'*(y_bin_list[x]==1.0)+'b'*(y_bin_list[x]==0.0)])
   return y_str	

def y_str2bin(y_str):
#  create 's' and 'b' columns from binary input   
    y_bin  = np.array([int(row[-1] == 's') for row in y_str[1:]])
    return y_bin


def  main():

        mode = 'g-mean-prob'
#_____________________________TRAINING SET_______________________________
   # Read training data
        y_trn_pred_prob,y_trn_pred,y_trn, weights,idx_trn=read_train_file() 
        n_train = 250000
        prob_threshold = 0.695
# Some probabilities of xgboost are negative. We make them positive and normalize on the maximum value of probability. Now all probabilities are in the region [0,1]. 
        y_trn_pred_prob[:,6] = abs(y_trn_pred_prob[:,6])
        y_trn_pred_prob_max = max(y_trn_pred_prob[:,6])

# Double check that we get the maximum element of xgboost probability.
        y_trn_pred_prob_max2 = 0
        for j in range(0,n_train):
            if y_trn_pred_prob[j,6]>y_trn_pred_prob_max2:
                y_trn_pred_prob_max2 = y_trn_pred_prob[j,6]
        if y_trn_pred_prob_max == y_trn_pred_prob_max2:
            print ' '
        else:
            print("2 methods of determing max xgboost prob-ty do not match..")
            input()     # To let the user see the error message
            sys.exit(1) # Exit the program
        y_trn_pred_prob[:,6] = y_trn_pred_prob[:,6]/y_trn_pred_prob_max

        prob_trn_arithmetic_mean = np.zeros(n_train)
        prob_trn_geometric_mean = np.zeros(n_train)

        for k in range(0,n_train):
            prob_trn_arithmetic_mean[k] = sum(y_trn_pred_prob[k,:])/len(names)
            prob_trn_geometric_mean[k] = 1
            for k1 in range(0,len(names)):
                prob_trn_geometric_mean[k] *= y_trn_pred_prob[k,k1] 
            prob_trn_geometric_mean[k] = (prob_trn_geometric_mean[k])**(1./len(names))

        y_trn_pred_logic=np.zeros(n_train)
        sum_trn_pred=np.zeros(n_train)

        sum_trn_pred_prob = np.zeros(n_train)
        # logic 'AND' or majority for every vector X -- TRAINING SET
        sum_trn_pred = np.sum(y_trn_pred, axis = 1)
      #  sum_trn_pred_prob = np.sum(y_trn_pred_prob, axis = 1)
      #  for k2 in range(0,n_train):
      #      prob_trn_geometric_mean[k2] = prob_trn_geometric_mean[k2]/sum_trn_pred_prob[k2] 

        for j in range(0,n_train):
           if mode == 'majority':
              if sum_trn_pred[j] >= (int(len(names))/2)+1:
                 y_trn_pred_logic[j] = 1
              else:
                 y_trn_pred_logic[j] = 0
           # arithmetic mean of probabilities greater than some threshold value.
           elif mode == 'a-mean-prob':
              if prob_trn_arithmetic_mean[j] > prob_threshold:
                 y_trn_pred_logic[j] = 1
              else:
                 y_trn_pred_logic[j] = 0
           elif mode == 'g-mean-prob':
              if prob_trn_geometric_mean[j] > prob_threshold:
                 y_trn_pred_logic[j] = 1
              else:
                 y_trn_pred_logic[j] = 0
           elif mode == 'xgboost':
                 y_trn_pred_logic[j] = y_trn_pred[j,len(names)-1]
           else:
              if sum_trn_pred[j] == len(names):
                 y_trn_pred_logic[j] = 1
              else:
                 y_trn_pred_logic[j] = 0
    # Calculate AMS
        truePos, falsePos = get_rates(y_trn_pred_logic, y_trn, weights)
        model_AMS = AMS(truePos, falsePos)

        print 'AMS_training =', model_AMS
#__________________________________________________________________________

#_______________________________TEST SET___________________________________
       # Read test data        
        y_pred_prob,y_pred,idx= read_test_file()       
        
        n_test = 550000
        y_pred_logic=np.zeros(n_test)
        sum_pred=np.zeros(n_test)

        y_pred_prob[:,6] = abs(y_pred_prob[:,6])
        y_pred_prob_max = max(y_pred_prob[:,6])
        y_pred_prob[:,6] = y_pred_prob[:,6]/y_pred_prob_max

        prob_arithmetic_mean = np.zeros(n_test)
        prob_geometric_mean = np.zeros(n_test)

        for k in range(0,n_test):
            prob_arithmetic_mean[k] = sum(y_pred_prob[k,:])/len(names)
            prob_geometric_mean[k] = 1
            for k1 in range(0,len(names)):
                prob_geometric_mean[k] *= y_pred_prob[k,k1]
            prob_geometric_mean[k] = (prob_geometric_mean[k])**(1./len(names))

        sum_pred = np.sum(y_pred, axis = 1)

        for j in range(n_test): 
           if mode == 'majority':
              if sum_pred[j] >= (int(len(names))/2)+1:
                 y_pred_logic[j] = 1
              else:
                 y_pred_logic[j] = 0
           # arithmetic mean of probabilities greater than some threshold value.
           elif mode == 'a-mean prob':
              if prob_arithmetic_mean[j] > prob_threshold:
                 y_pred_logic[j] = 1
              else:
                 y_pred_logic[j] = 0
           elif mode == 'g-mean-prob':
              if prob_geometric_mean[j] > prob_threshold:
                 y_pred_logic[j] = 1
              else:
                 y_pred_logic[j] = 0
           elif mode == 'xgboost':
                 y_pred_logic[j] = y_pred[j,len(names)-1]
           else:
              if sum_pred[j] == len(names):
                 y_pred_logic[j] = 1   
              else:
                 y_pred_logic[j] = 0

# the name of final output file
        final_name='final_higgs_'+mode      
        fname=final_name+'_test.csv'

        #y_pred_logic=np.zeros(n_test)        
        order = range(1, n_test+1)
        # make rorder for signal be on the top 
        save_results_final(idx,order,y_pred_logic,fname)
#_________________________________________________________________
 
if __name__ == "__main__":
    main()
