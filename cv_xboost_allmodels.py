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
#    print('Writing a final csv for %s'% name_method)  
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


def estimate_performance_xgboost(param, num_round, folds):
    '''
    Cross validation for XGBoost performance 
    '''
    # Load training data   
#    y_pred_prob,y_pred,y,weights,idx=get_pred_file('train') 
    y_pred_prob,y_pred,y, weights,idx=read_train_file() 
    X=y_pred_prob
    labels=y

#    X, labels, weights = get_training_data(training_file)
    # Cross validate
    kf = cv.KFold(y.size, n_folds=folds)
#    npoints  = 6
    npoints  = 10
    # Dictionary to store all the AMSs
    all_AMS = {}
    for curr in range(npoints):
        all_AMS[curr] = []
    # These are the cutoffs used for the XGBoost predictions
    cutoffs  = sp.linspace(0.05, 0.50, npoints)
    for train_indices, test_indices in kf:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]
        w_train, w_test = weights[train_indices], weights[test_indices]

        # Rescale weights so that their sum is the same as for the entire training set
        w_train *= (sum(weights) / sum(w_train))
        w_test  *= (sum(weights) / sum(w_test))

        sum_wpos = sum(w_train[y_train == 1])
        sum_wneg = sum(w_train[y_train == 0])

        # construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
        xgmat = xgb.DMatrix(X_train, label=y_train, missing=-999.0, weight=w_train)

        # scale weight of positive examples
        param['scale_pos_weight'] = sum_wneg / sum_wpos
        # you can directly throw param in, though we want to watch multiple metrics here 
        plst = param.items()#+[('eval_metric', 'ams@0.15')]

        watchlist = []#[(xgmat, 'train')]
        bst = xgb.train(plst, xgmat, num_round, watchlist)

        # Construct matrix for test set
        xgmat_test = xgb.DMatrix(X_test, missing=-999.0)
        y_out = bst.predict(xgmat_test)
        res  = [(i, y_out[i]) for i in xrange(len(y_out))]
        rorder = {}
        for k, v in sorted(res, key = lambda x:-x[1]):
            rorder[k] = len(rorder) + 1

        # Explore changing threshold_ratio and compute AMS
        best_AMS = -1.
        for curr, threshold_ratio in enumerate(cutoffs):
            y_pred = sp.zeros(len(y_out))
            ntop = int(threshold_ratio * len(rorder))
            for k, v in res:
                if rorder[k] <= ntop:
                    y_pred[k] = 1

            truePos, falsePos = get_rates(y_pred, y_test, w_test)
            this_AMS = AMS(truePos, falsePos)
            all_AMS[curr].append(this_AMS)
            if this_AMS > best_AMS:
                best_AMS = this_AMS
        print("Best AMS with trhs:%g,num_round:%g,mx_depth:%g =%g"%(threshold_ratio,num_round,max_depth,best_AMS))
    print "------------------------------------------------------"
    outfile = './result_train_all/xgboost_cv_all_nr_%d_md_%d.csv'%(int(num_round),int(max_depth))
    fo = open(outfile, 'w')
    for curr, cut in enumerate(cutoffs):
#        print "Thresh = %.2f: AMS = %.4f, std = %.4f" % \
#            (cut, sp.mean(all_AMS[curr]), sp.std(all_AMS[curr]))
        print("Num_round=%.1f,Max_Depth=%.1f,Thresh = %.2f: AMS = %.4f, std = %.4f" % \
            (num_round ,max_depth,cut, sp.mean(all_AMS[curr]), sp.std(all_AMS[curr])))
        fo.write("Num_round=%.1f,Max_Depth=%.1f,Thresh = %.2f: AMS = %.4f, std = %.4f" % \
            (num_round ,max_depth,cut, sp.mean(all_AMS[curr]), sp.std(all_AMS[curr])))

    print "------------------------------------------------------"
    fo.close() 


def  calculate_final_performance_xgboost(max_depth=6, num_round=120,threshold_ratio=0.15):
     
       # Load training data
      
        y_pred_prob,y_pred,y, weight,idx=read_train_file() 
	        
        param = {}        
        param['objective'] = 'binary:logitraw'
        param['bst:eta'] = 0.1 
        param['bst:max_depth'] =max_depth
        param['eval_metric'] = 'auc'
        param['silent'] = 1
        param['nthread'] = 8
        # scale weight of positive examples
        sum_wpos = sum( weight[i] for i in range(len(y)) if y[i] == 1.0  )
        sum_wneg = sum( weight[i] for i in range(len(y)) if y[i] == 0.0  )
        param['scale_pos_weight'] = sum_wneg / sum_wpos
      
          
        # construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
        X=y_pred_prob
        xgmat = xgb.DMatrix(X,label=y, missing=-999.0, weight=weight)

        # you can directly throw param in, though we want to watch multiple metrics here 
        plst = param.items()+[('eval_metric', 'ams@0.15')]

#        watchlist = []#[(xgmat, 'train')]
	watchlist = [ (xgmat,'train') ]
        bst = xgb.train(plst, xgmat, num_round, watchlist)     
#       read train data        
        y_pred_prob,y_pred,idx= read_test_file()       
        X_test=y_pred_prob
        # construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
        xgmat = xgb.DMatrix(X_test, missing = -999.0 )
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

        outdir='./final_test_csv/'         
        name='final_all_max_depth=%i_num_round=%i_thold=%i'%(max_depth,num_round,int(100*threshold_ratio))      
        fname=outdir+name+'_test.csv'
#        save_results_final(idx,range(1,len(idx)+1),y_pred,fname)
        #  transform rorder from dict to array  
        rorder_array=np.array(rorder.items())[:,1]
        # make rorder for signal be on the top 
        save_results_final(idx,len(rorder_array)+1-rorder_array,y_pred,fname)

 
def main():

    global num_round  # Number of boosted trees
    global max_depth
    # setup parameters for xgboost
    mode='test'
    if mode=='train':
     param = {}
    # use logistic regression loss, use raw prediction before logistic transformation
    # since we only need the rank
     param['objective'] = 'binary:logitraw'
     param['bst:eta'] = 0.1    
     param['eval_metric'] = 'auc'
     param['silent'] = 1
     param['nthread'] = 8
    
     folds = 3 # Folds for CV
     numRound=[5, 10, 15,20, 60, 120, 240]
     maxDepth=[1,2,3,4,5,6,7,8,10, 15]
     for i in range(len(numRound)):
      for k in range(len(maxDepth)):
        num_round = numRound[i] # Number of boosted trees
        max_depth=maxDepth[k]
        param['bst:max_depth'] = max_depth
        estimate_performance_xgboost( param, num_round, folds)
    elif mode=='test':
     max_depth=4
     num_round=60
     threshold_ratio=0.15
     calculate_final_performance_xgboost(max_depth, num_round,threshold_ratio)

if __name__ == "__main__":
    main()
