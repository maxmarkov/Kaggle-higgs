Folder "predansam_our_development_and_results" contains the python scripts (cv_xgboost_allmodels.py and logic_allmodels_v1.py) which apply xgboost algorithl and some simple logical (logical "AND" or "OR")
 selection on the probabilities calculated with the set of approaches such as xgboost (the most 
powerful) and some built-in sklearn algorithms like xboost, random forest (RF), K-nearest neighbor 
algorithm (KNN), gradient boosting classifier (GBC), linear support vector machines (LSVM), support 
vector machine with rbf kernel (RBFSVM1), Linear Discriminant Analysis (LDA). 


The idea is to combine the strengths of the different algorithms to produce the most
 reliable and exact selection of the events. Unfortunately, it is just slightly increase the result of
 the most powerful xgboosti algorithm in the best combination. 

Dataout contains the results of the trainings of the algorithms.
Datain contains input files for the Higgs boson competition.
Final_test_csv contains some results of the logical selection algorithm.
cv_xgboost_allmodels.py - applies xgboost to probabilities.
logic_allmodels_v1.py - applies logical selection to probabilities.
results.txt - contains the results of the different machine learning algorithms themselves (without any secondary probability learning).
