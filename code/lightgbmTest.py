import numpy as np
from lightgbm import LGBMClassifier
from lightgbm import early_stopping
from lightgbm import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

import os
os.environ["PATH"] += os.pathsep + 'D:\python\Graphviz\\bin'

traindata = pd.read_csv("data\\traindata.csv")
trainlabel = np.genfromtxt("data\\trainlabel.txt",dtype=int)
trainlabel = np.array(trainlabel)

traindata['workclass'] = traindata['workclass'].astype('category')
traindata['education'] = traindata['education'].astype('category')
traindata['marital.status'] = traindata['marital.status'].astype('category')
traindata['occupation'] = traindata['occupation'].astype('category')
traindata['relationship'] = traindata['relationship'].astype('category')
traindata['race'] = traindata['race'].astype('category')
traindata['sex'] = traindata['sex'].astype('category')
traindata['native.country'] = traindata['native.country'].astype('category')


# bulid a basic lightgbm model
# cf = LGBMClassifier(max_depth=3)
# cf.fit(traindata,trainlabel,categorical_feature=["workclass","education","marital.status","occupation","relationship","race","sex","native.country"])
# print("调参前：", accuracy_score(trainlabel, cf.predict(traindata)))

# adjust the parameters 

# learning_rate:
# ScoreAll = []
# for i in [0.05,0.06,0.07,0.08,0.09,0.1]:
#     cf = LGBMClassifier(max_depth=3,learning_rate=i)
#     cf.fit(traindata,trainlabel,categorical_feature=["workclass","education","marital.status","occupation","relationship","race","sex","native.country"])
#     score = accuracy_score(trainlabel, cf.predict(traindata))
#     ScoreAll.append([i,score])
# ScoreAll = np.array(ScoreAll)

# max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] ##这句话看似很长的，其实就是找出最高得分对应的索引
# print("最优learning_rate以及最高得分:",ScoreAll[max_score])  
# plt.figure(figsize=[20,5])
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.xlabel('learning_rate')
# plt.ylabel('accuracy')
# plt.show()

# min_child_sample
# ScoreAll = []
# for i in range(20,100,10):
#     cf = LGBMClassifier(max_depth=3,learning_rate=0.1, min_child_samples=i)
#     cf.fit(traindata,trainlabel,categorical_feature=["workclass","education","marital.status","occupation","relationship","race","sex","native.country"])
#     score = accuracy_score(trainlabel, cf.predict(traindata))
#     ScoreAll.append([i,score])
# ScoreAll = np.array(ScoreAll)

# max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] ##这句话看似很长的，其实就是找出最高得分对应的索引
# print("最优min_child_sample以及最高得分:",ScoreAll[max_score])  
# plt.figure(figsize=[20,5])
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.xlabel('min_child_sample')
# plt.ylabel('accuracy')
# plt.show()

# max_depth
# ScoreAll = []
# for i in range(3,6):
#     cf = LGBMClassifier(max_depth=i,learning_rate=0.1, min_child_samples=60)
#     cf.fit(traindata,trainlabel,categorical_feature=["workclass","education","marital.status","occupation","relationship","race","sex","native.country"])
#     score = accuracy_score(trainlabel, cf.predict(traindata))
#     ScoreAll.append([i,score])
# ScoreAll = np.array(ScoreAll)

# max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] ##这句话看似很长的，其实就是找出最高得分对应的索引
# print("最优max_depth以及最高得分:",ScoreAll[max_score])  
# plt.figure(figsize=[20,5])
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.xlabel('max_depth')
# plt.ylabel('accuracy')
# plt.show()

# num_leaves
# ScoreAll = []
# for i in range(2,255,8):
#     cf = LGBMClassifier(max_depth=5,learning_rate=0.1, min_child_samples=60, num_leaves=i)
#     cf.fit(traindata,trainlabel,categorical_feature=["workclass","education","marital.status","occupation","relationship","race","sex","native.country"])
#     score = accuracy_score(trainlabel, cf.predict(traindata))
#     ScoreAll.append([i,score])
# ScoreAll = np.array(ScoreAll)

# max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] ##这句话看似很长的，其实就是找出最高得分对应的索引
# print("最优num_leaves以及最高得分:",ScoreAll[max_score])  
# plt.figure(figsize=[20,5])
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.xlabel('num_leaves')
# plt.ylabel('accuracy')
# plt.show()

# subsample
# ScoreAll = []
# for i in [0.8,0.85,0.9,0.95,1]:
#     cf = LGBMClassifier(max_depth=5,learning_rate=0.1, min_child_samples=60, num_leaves=90, subsample=i)
#     cf.fit(traindata,trainlabel,categorical_feature=["workclass","education","marital.status","occupation","relationship","race","sex","native.country"])
#     score = accuracy_score(trainlabel, cf.predict(traindata))
#     ScoreAll.append([i,score])
# ScoreAll = np.array(ScoreAll)

# max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] ##这句话看似很长的，其实就是找出最高得分对应的索引
# print("最优subsample以及最高得分:",ScoreAll[max_score])  
# plt.figure(figsize=[20,5])
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.xlabel('subsample')
# plt.ylabel('accuracy')
# plt.show()


# colsample_bytree
# ScoreAll = []
# for i in [0.8,0.85,0.9,0.95,1]:
#     cf = LGBMClassifier(max_depth=5,learning_rate=0.1, min_child_samples=60, num_leaves=90, subsample=0.8, colsample_bytree= i)
#     cf.fit(traindata,trainlabel,categorical_feature=["workclass","education","marital.status","occupation","relationship","race","sex","native.country"])
#     score = accuracy_score(trainlabel, cf.predict(traindata))
#     ScoreAll.append([i,score])
# ScoreAll = np.array(ScoreAll)

# max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] ##这句话看似很长的，其实就是找出最高得分对应的索引
# print("最优colsample_bytree以及最高得分:",ScoreAll[max_score])  
# plt.figure(figsize=[20,5])
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.xlabel('colsample_bytree')
# plt.ylabel('accuracy')
# plt.show()


# param_grid = {
#     'max_depth':np.arange(3,6),
#     'min_child_samples':np.arange(40, 70, 10),
#     'num_leaves':np.arange(70, 106, 5),
#     'subsample' : [0.8,0.9,1],
#     'colsample_bytree' : [0.8,0.9,1]
#     }

# rfc = LGBMClassifier()
# GS = GridSearchCV(rfc,param_grid,cv=10)
# GS.fit(traindata,trainlabel)
# print(GS.best_params_)
# print(GS.best_score_)

cf = LGBMClassifier(learning_rate=0.1, max_depth=5, min_child_samples=50, num_leaves=70, subsample=0.8, colsample_bytree=0.8, n_estimators=300)
cf.fit(traindata,trainlabel,categorical_feature=["workclass","education","marital.status","occupation","relationship","race","sex","native.country"])
# #print("调参后：", accuracy_score(trainlabel, cf.predict(traindata)))

# fig2 = plt.figure(figsize=(20, 20))
# ax = fig2.subplots()
# plot_tree(cf, tree_index=1, ax=ax)
# plt.show()

# predict testdata
testdata = pd.read_csv("data\\testdata.csv")
testdata.replace('Holand-Netherlands', '?', inplace=True)
testdata['workclass'] = testdata['workclass'].astype('category')
testdata['education'] = testdata['education'].astype('category')
testdata['marital.status'] = testdata['marital.status'].astype('category')
testdata['occupation'] = testdata['occupation'].astype('category')
testdata['relationship'] = testdata['relationship'].astype('category')
testdata['race'] = testdata['race'].astype('category')
testdata['sex'] = testdata['sex'].astype('category')
testdata['native.country'] = testdata['native.country'].astype('category')

pred = cf.predict(testdata)
print(pred)
# write in txt
f=open("predict/testlabel.txt","w")
 
for line in pred:
    f.write(str(line)+'\n')
f.close()








