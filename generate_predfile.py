#import ipdb; ipdb.set_trace()

import csv
import sys
import numpy as np
import scipy as sp
import pandas as pd
import math
import os
import inspect

# add path of xgboost python module
code_path = os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0], "../../../python")
sys.path.append(code_path)

import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

from sklearn import grid_search
import sklearn.cross_validation as cv
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.externals import joblib

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

outdir='./dataout/'
training_file='./datain/training.csv'
test_file='./datain/test.csv'

names = ["KNN", "RF", "GBC"]
#names = []
classifiers = [
    KNeighborsClassifier(50,algorithm='auto',p=1),
    RandomForestClassifier(max_depth=15, n_estimators=120, n_jobs=-1),
    GBC(n_estimators=120, max_depth=6,min_samples_leaf=200,max_features=10,verbose=1)]     
#    GBM(n_estimators=5, max_depth=5,min_samples_leaf=200,max_features=10,verbose=1) 
#   GBM(n_estimators=150, max_depth=5,min_samples_leaf=200,max_features=10,subsample=0.5,sample_weight=weights,verbose=1) 
#    KNeighborsClassifier(50,weights=weights,algorithm='auto',p=1),

names_norm = ["LSVM","RBFSVM1","LDA"]

classifiers_norm = [
    SVC(kernel="linear",C=0.025,cache_size=1000,probability=True),
    SVC(kernel="rbf",C=1,gamma=0.03,cache_size=1000,probability=True),
    LDA()]

#names_norm = ["LDA"]
#classifiers_norm = [LDA()]


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

def get_training_data(training_file):
    '''
    Loads training data.
    '''
    data = list(csv.reader(open(training_file, "rb"), delimiter=','))
    X       = np.array([map(float, row[1:-2]) for row in data[1:]])
    y  = np.array([int(row[-1] == 's') for row in data[1:]])
    weights = np.array([float(row[-2]) for row in data[1:]])
    idx     = np.array([int(row[0]) for row in data[1:]])
    return X, y, weights,idx
#    return X[:500,:], y[:500], weights[:500],idx[:500]


def get_test_data(training_file):
    '''
    Loads test data.
    '''
    data = list(csv.reader(open(training_file, "rb"), delimiter=','))
    X       = np.array([map(float, row[1:]) for row in data[1:]])
#    y  = np.array([int(row[-1] == 's') for row in data[1:]])
#    weights = np.array([float(row[-2]) for row in data[1:]])
    idx     = np.array([int(row[0]) for row in data[1:]])
    return X, idx
#    return X[:500,:], y[:500], weights[:500],idx[:500]

def save_results_learn(idx,y_pred_prob,y_pred,weights,y,fname):    
    # Write the result list data to a csv file
#    print('Writing a final csv for %s'% name_method)
    df = pd.DataFrame({"EventId": idx, "YProb": y_pred_prob,"YPred":y_pred,'W':weights,"Y":y})
    df.to_csv(fname, index=False, cols=["EventId", "YProb","YPred","W","Y",])

def save_results_final(idx,y_pred_prob,y_pred,fname):    
    # Write the result list data to a csv file
#    print('Writing a final csv for %s'% name_method)  
    y_str=y_bin2str(y_pred)
    df = pd.DataFrame({"EventId": idx, "RankOrder": range(1,len(idx)+1),"Class":y_str})
    df.to_csv(fname, index=False, cols=["EventId", "RankOrder", "Class"])

def save_results_final_pre(idx,y_prob,y_pred,fname):
    # Write the result list data to a csv file
#    print('Writing a final csv for %s'% name_method)  
    y_str=y_bin2str(y_pred)
    df = pd.DataFrame({"EventId": idx, "YProb": y_prob,"YBin":y_pred,"YStr":y_str})
    df.to_csv(fname, index=False, cols=["EventId", "YProb", "YBin","YStr"])


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

def get_performance_all(train_model_mode=0):
    print( 'train_model_mode=%i'%train_model_mode)

    '''
    we generate the files with probability and classification trees
    '''   
    # Load training data
# save output of non-normalized classfiers   
    X, y, weights,idx = get_training_data(training_file)
#    X, y, weights,idx=X[:1000,:], y[:1000], weights[:1000],idx[:1000]
    X_test, idx_test = get_test_data(test_file)
#    X_test, idx_test =X[:1000,:], idx_test[:1000]

    for name, clf in zip(names, classifiers):
        print name
        fname=outdir+name+'_model.pkl'
     
        if train_model_mode:
         clf.fit(X, y)   
         joblib.dump(clf,fname) 
        else:
         print('loading model from %s'%fname)
         clf=joblib.load(fname)

        y_pred_prob= clf.predict_proba(X)[:,1]      
        y_pred=clf.predict(X)
#        print len(y_pred_prob),len(y_pred),len(idx)
        fname=outdir+name+'_train.csv'
        save_results_learn(idx,y_pred_prob,y_pred,weights,y,fname)

        y_pred_prob= clf.predict_proba(X_test)[:,1]           
        y_pred=clf.predict(X_test)
#        print len(y_pred_prob),len(y_pred),len(idx_test)
        fname=outdir+name+'_test.csv'
        save_results_final(idx_test,y_pred_prob,y_pred,fname)
        fname=outdir+name+'_test_pre.csv'
        save_results_final_pre(idx_test,y_pred_prob,y_pred,fname)


    X, y, weights,idx = get_training_data(training_file)
#    X, y, weights,idx=X[:1000,:], y[:1000], weights[:1000],idx[:1000]
    imp=Imputer(missing_values=-999.0, strategy='mean',axis=0)
    X=imp.fit_transform(X)
    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    X_test, idx_test = get_test_data(test_file)
#    X_test, idx_test =X_test[:1000,:], idx_test[:1000]
    imp=Imputer(missing_values=-999.0, strategy='mean',axis=0)
    X_test=imp.fit_transform(X_test)
    scaler=StandardScaler()
    X_test=scaler.fit_transform(X_test)

    for name, clf in zip(names_norm, classifiers_norm):
        print name
        fname=outdir+name+'_model.pkl'
        if train_model_mode:
         clf.fit(X, y)    
         joblib.dump(clf,fname)
        else:
         print('loading model from %s'%fname)
         clf=joblib.load(fname)
 
        y_pred_prob= clf.predict_proba(X)[:,1]
        y_pred=clf.predict(X)
        fname=outdir+name+'_train.csv'
        save_results_learn(idx,y_pred_prob,y_pred,weights,y,fname)

        y_pred_prob= clf.predict_proba(X_test)[:,1]    
        y_pred=clf.predict(X_test)
        fname=outdir+name+'_test.csv'
        save_results_final(idx_test,y_pred_prob,y_pred,fname)
        fname=outdir+name+'_test_pre.csv'
        save_results_final_pre(idx_test,y_pred_prob,y_pred,fname)
          
def get_performance_xgboost():    
	'''
    we generate the files with probability and classification trees
	'''
        param = {}
        # use logistic regression loss, use raw prediction before logistic transformation
        # since we only need the rank
        param['objective'] = 'binary:logitraw'
        # scale weight of positive examples

        param['bst:eta'] = 0.1 
        param['bst:max_depth'] = 6
        param['eval_metric'] = 'auc'
        param['silent'] = 1
        param['nthread'] = 8
       # boost 120 tres
        num_round=120
        threshold_ratio=0.15
       # Load training data
        X, y, weight,idx = get_training_data(training_file)        
        # construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
        xgmat = xgb.DMatrix(X,label=y, missing=-999.0, weight=weight)

        # scale weight of positive examples
        sum_wpos = sum( weight[i] for i in range(len(y)) if y[i] == 1.0  )
        sum_wneg = sum( weight[i] for i in range(len(y)) if y[i] == 0.0  )
        param['scale_pos_weight'] = sum_wneg / sum_wpos
        # you can directly throw param in, though we want to watch multiple metrics here 
#        plst = param.items()+[('eval_metric', 'ams@0.15')]
        plst = param.items()

#        watchlist = []#[(xgmat, 'train')]
	watchlist = [ (xgmat,'train') ]
        bst = xgb.train(plst, xgmat, num_round, watchlist)     
        y_out = bst.predict(xgmat)
        y_pred = sp.zeros(len(y_out)) 
        res  = [(i, y_out[i]) for i in xrange(len(y_out))]
        rorder = {}
        for k, v in sorted(res, key = lambda x:-x[1]):
            rorder[k] = len(rorder) + 1

        ntop = int(threshold_ratio * len(rorder))
        for k, v in res:
         if rorder[k] <= ntop:
              y_pred[k] = 1

        name='xboost'         
        fname=outdir+name+'_train.csv'
        save_results_learn(idx,y_out,y_pred,weight,y,fname)
     
     
        X_test, idx_test = get_test_data(test_file)       
        # construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
        xgmat = xgb.DMatrix( X_test, missing = -999.0 )
        y_out = bst.predict(xgmat)
        y_pred = sp.zeros(len(y_out)) 
        res  = [(i, y_out[i]) for i in xrange(len(y_out))]
        rorder = {}
        for k, v in sorted(res, key = lambda x:-x[1]):
            rorder[k] = len(rorder) + 1

        ntop = int(threshold_ratio * len(rorder))
        for k, v in res:
         if rorder[k] <= ntop:
              y_pred[k] = 1    
        fname=outdir+name+'_test.csv'
        save_results_final(idx_test,range(1,len(idx_test)+1),y_pred,fname)
        fname=outdir+name+'_test_pre.csv'
        save_results_final_pre(idx_test,y_out,y_pred,fname)
 

def main():
    train_model_mode=0
    get_performance_all(train_model_mode)    
    get_performance_xgboost()

#        estimate_performance_xgboost("../data/training.csv",num_round, max_depth, folds)


if __name__ == "__main__":
    main()
