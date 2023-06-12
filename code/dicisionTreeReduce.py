#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

# input data
traindata = pd.read_csv("data\\traindata.csv")
traindata['workclass'].replace('?', 'unknown_workclass',inplace=True)
traindata['occupation'].replace('?', 'unknown_occupation',inplace=True)
traindata['native.country'].replace('?', 'unknown_country',inplace=True)

workclass = traindata['workclass']
education = traindata['education']
marital_status = traindata['marital.status']
occupation = traindata['occupation']
relationship = traindata['relationship']
race = traindata['race']
sex = traindata['sex']
native_country = traindata['native.country']

workclass = np.array(workclass)
education = np.array(education)
marital_status = np.array(marital_status)
occupation = np.array(occupation)
relationship = np.array(relationship)
race = np.array(race)
sex = np.array(sex)
native_country = np.array(native_country)

# one-hot encoding
oe = OneHotEncoder()
oe.fit(workclass.reshape(-1, 1))
encoded_labels = oe.transform(workclass.reshape(-1, 1)).toarray()
encoded_labels_df = pd.DataFrame(encoded_labels, columns=oe.categories_[0])
traindata.drop('workclass',axis=1, inplace=True)
traindata = traindata.join(encoded_labels_df)

oe.fit(education.reshape(-1, 1))
encoded_labels = oe.transform(education.reshape(-1, 1)).toarray()
encoded_labels_df = pd.DataFrame(encoded_labels, columns=oe.categories_[0])
traindata.drop('education',axis=1, inplace=True)
traindata = traindata.join(encoded_labels_df)

oe.fit(marital_status.reshape(-1, 1))
encoded_labels = oe.transform(marital_status.reshape(-1, 1)).toarray()
encoded_labels_df = pd.DataFrame(encoded_labels, columns=oe.categories_[0])
traindata.drop('marital.status',axis=1, inplace=True)
traindata = traindata.join(encoded_labels_df)

oe.fit(occupation.reshape(-1, 1))
encoded_labels = oe.transform(occupation.reshape(-1, 1)).toarray()
encoded_labels_df = pd.DataFrame(encoded_labels, columns=oe.categories_[0])
traindata.drop('occupation',axis=1, inplace=True)
traindata = traindata.join(encoded_labels_df)

oe.fit(relationship.reshape(-1, 1))
encoded_labels = oe.transform(relationship.reshape(-1, 1)).toarray()
encoded_labels_df = pd.DataFrame(encoded_labels, columns=oe.categories_[0])
traindata.drop('relationship',axis=1, inplace=True)
traindata = traindata.join(encoded_labels_df)

oe.fit(race.reshape(-1, 1))
encoded_labels = oe.transform(race.reshape(-1, 1)).toarray()
encoded_labels_df = pd.DataFrame(encoded_labels, columns=oe.categories_[0])
traindata.drop('race',axis=1, inplace=True)
traindata = traindata.join(encoded_labels_df)

oe.fit(sex.reshape(-1, 1))
encoded_labels = oe.transform(sex.reshape(-1, 1)).toarray()
encoded_labels_df = pd.DataFrame(encoded_labels, columns=oe.categories_[0])
traindata.drop('sex',axis=1, inplace=True)
traindata = traindata.join(encoded_labels_df)

oe.fit(native_country.reshape(-1, 1))
encoded_labels = oe.transform(native_country.reshape(-1, 1)).toarray()
encoded_labels_df = pd.DataFrame(encoded_labels, columns=oe.categories_[0])
traindata.drop('native.country',axis=1, inplace=True)
traindata = traindata.join(encoded_labels_df)

# reduce the demension
selectFeatures = ['education.num', 'capital.gain', 'age', 'capital.loss', 'hours.per.week', 'fnlwgt', 'Self-emp-not-inc','Federal-gov','Private',
                  'Self-emp-inc','Local-gov','State-gov']
traindata = pd.DataFrame(traindata, columns=selectFeatures)

traindata = np.array(traindata)
trainlabel = np.genfromtxt("data\\trainlabel.txt",dtype=int)
trainlabel = np.array(trainlabel)

train,test,train_label,test_label = train_test_split(traindata,trainlabel,test_size=0.1,random_state=0)

# #build a basic entropy decisionTree
# DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 66)
# DT.fit(train, train_label)
# pred1 = DT.predict(test)
# accuracy = accuracy_score(test_label,pred1)
# print("调参前accuracy:", accuracy)

# #adjust the tree
# #max_depth
# ScoreAll = []
# for i in range(1,100,10):
#     DT = DecisionTreeClassifier(criterion = 'entropy', max_depth = i,random_state = 66)
#     score = cross_val_score(DT,traindata,trainlabel,cv=10).mean()
#     ScoreAll.append([i,score])
# ScoreAll = np.array(ScoreAll)

# max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] ##这句话看似很长的，其实就是找出最高得分对应的索引
# print("最优max_depth以及最高得分:",ScoreAll[max_score])  
# plt.figure(figsize=[20,5])
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.xlabel('max depth')
# plt.ylabel('cross_val_score')
# plt.show()

# #min_samples_split
# ScoreAll = []
# for i in range(15,50):
#     DT = DecisionTreeClassifier(criterion = 'entropy', max_depth = 8,min_samples_split = i,random_state = 66)
#     score = cross_val_score(DT,traindata,trainlabel,cv=10).mean()
#     ScoreAll.append([i,score])
# ScoreAll = np.array(ScoreAll)

# max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] ##这句话看似很长的，其实就是找出最高得分对应的索引
# print("最优min_samples_split以及最高得分:",ScoreAll[max_score])  
# plt.figure(figsize=[20,5])
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.xlabel('min_samples_split')
# plt.ylabel('cross_val_score')
# plt.show()

# #min_samples_leaf
# ScoreAll = []
# for i in range(20,50):
#     DT = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = i,min_samples_split = 32,max_depth = 8,random_state = 66)
#     score = cross_val_score(DT,traindata,trainlabel,cv=10).mean()
#     ScoreAll.append([i,score])
# ScoreAll = np.array(ScoreAll)

# max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] ##这句话看似很长的，其实就是找出最高得分对应的索引
# print("最优min_samples_leaf以及最高得分:",ScoreAll[max_score])  
# plt.figure(figsize=[20,5])
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.xlabel('min_samples_leaf')
# plt.ylabel('cross_val_score')
# plt.show()

# # adjust max_depth,min_samples_leaf and min_samples_split together
# param_grid = {
#     'max_depth':np.arange(5, 13),
#     'min_samples_leaf':np.arange(28, 35),
#     'min_samples_split':np.arange(25, 32)}

# rfc = DecisionTreeClassifier(criterion='entropy', random_state=66)
# GS = GridSearchCV(rfc,param_grid,cv=10)
# GS.fit(traindata,trainlabel)
# print(GS.best_params_)
# print(GS.best_score_)


# DT = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 30, min_samples_split = 25, max_depth = 11, random_state = 66)
# DT.fit(train, train_label)
# pred1 = DT.predict(test)
# accuracy = accuracy_score(test_label,pred1)
# print("调参后accuracy:", accuracy)

# plt.figure(figsize=(15,9))
# plot_tree(DT,filled=True)
# plt.show()



# build a basic CART decisionTree
DT = DecisionTreeClassifier(random_state = 66)
DT.fit(train, train_label)
pred1 = DT.predict(test)
accuracy = accuracy_score(test_label,pred1)
print("调参前accuracy:", accuracy)

# adjust the tree
# max_depth
# ScoreAll = []
# for i in range(1,15,1):
#     DT = DecisionTreeClassifier(max_depth = i,random_state = 66)
#     score = cross_val_score(DT,traindata,trainlabel,cv=10).mean()
#     ScoreAll.append([i,score])
# ScoreAll = np.array(ScoreAll)

# max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] ##这句话看似很长的，其实就是找出最高得分对应的索引
# print("最优max_depth以及最高得分:",ScoreAll[max_score])  
# plt.figure(figsize=[20,5])
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.xlabel('max depth')
# plt.ylabel('cross_val_score')
# plt.show()

#min_samples_split
# ScoreAll = []
# for i in range(5,30):
#     DT = DecisionTreeClassifier(max_depth = 9,min_samples_split = i,random_state = 66)
#     score = cross_val_score(DT,traindata,trainlabel,cv=10).mean()
#     ScoreAll.append([i,score])
# ScoreAll = np.array(ScoreAll)

# max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] ##这句话看似很长的，其实就是找出最高得分对应的索引
# print("最优min_samples_split以及最高得分:",ScoreAll[max_score])  
# plt.figure(figsize=[20,5])
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.xlabel('min_samples_split')
# plt.ylabel('cross_val_score')
# plt.show()

#min_samples_leaf
# ScoreAll = []
# for i in range(10,30):
#     DT = DecisionTreeClassifier(min_samples_leaf = i,min_samples_split = 15,max_depth = 9,random_state = 66)
#     score = cross_val_score(DT,traindata,trainlabel,cv=10).mean()
#     ScoreAll.append([i,score])
# ScoreAll = np.array(ScoreAll)

# max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] ##这句话看似很长的，其实就是找出最高得分对应的索引
# print("最优min_samples_leaf以及最高得分:",ScoreAll[max_score])  
# plt.figure(figsize=[20,5])
# plt.plot(ScoreAll[:,0],ScoreAll[:,1])
# plt.xlabel('min_samples_leaf')
# plt.ylabel('cross_val_score')
# plt.show()

# #adjust max_depth,min_samples_leaf and min_samples_split together
# param_grid = {
#     'max_depth':np.arange(7, 11),
#     'min_samples_leaf':np.arange(8, 16),
#     'min_samples_split':np.arange(10, 20)}

# rfc = DecisionTreeClassifier(random_state=66)
# GS = GridSearchCV(rfc,param_grid,cv=10)
# GS.fit(traindata,trainlabel)
# print(GS.best_params_)
# print(GS.best_score_)


DT = DecisionTreeClassifier(min_samples_leaf = 9, min_samples_split = 10, max_depth = 9, random_state = 66)
DT.fit(train, train_label)
pred1 = DT.predict(test)
accuracy = accuracy_score(test_label,pred1)
print("调参后accuracy:", accuracy)

plt.figure(figsize=(15,9))
plot_tree(DT,filled=True)
plt.show()


# # predict the testdata
# testdata = pd.read_csv("data\\testdata.csv")
# testdata.replace('Holand-Netherlands', '?', inplace=True)

# testdata['workclass'].replace('?', 'unknown_workclass',inplace=True)
# testdata['occupation'].replace('?', 'unknown_occupation',inplace=True)
# testdata['native.country'].replace('?', 'unknown_country',inplace=True)

# workclass = testdata['workclass']
# education = testdata['education']
# marital_status = testdata['marital.status']
# occupation = testdata['occupation']
# relationship = testdata['relationship']
# race = testdata['race']
# sex = testdata['sex']
# native_country = testdata['native.country']

# workclass = np.array(workclass)
# education = np.array(education)
# marital_status = np.array(marital_status)
# occupation = np.array(occupation)
# relationship = np.array(relationship)
# race = np.array(race)
# sex = np.array(sex)
# native_country = np.array(native_country)

# # one-hot encoding
# oe = OneHotEncoder()
# oe.fit(workclass.reshape(-1, 1))
# encoded_labels = oe.transform(workclass.reshape(-1, 1)).toarray()
# encoded_labels_df = pd.DataFrame(encoded_labels, columns=oe.categories_[0])
# testdata.drop('workclass',axis=1, inplace=True)
# testdata = testdata.join(encoded_labels_df)

# oe.fit(education.reshape(-1, 1))
# encoded_labels = oe.transform(education.reshape(-1, 1)).toarray()
# encoded_labels_df = pd.DataFrame(encoded_labels, columns=oe.categories_[0])
# testdata.drop('education',axis=1, inplace=True)
# testdata = testdata.join(encoded_labels_df)

# oe.fit(marital_status.reshape(-1, 1))
# encoded_labels = oe.transform(marital_status.reshape(-1, 1)).toarray()
# encoded_labels_df = pd.DataFrame(encoded_labels, columns=oe.categories_[0])
# testdata.drop('marital.status',axis=1, inplace=True)
# testdata = testdata.join(encoded_labels_df)

# oe.fit(occupation.reshape(-1, 1))
# encoded_labels = oe.transform(occupation.reshape(-1, 1)).toarray()
# encoded_labels_df = pd.DataFrame(encoded_labels, columns=oe.categories_[0])
# testdata.drop('occupation',axis=1, inplace=True)
# testdata = testdata.join(encoded_labels_df)

# oe.fit(relationship.reshape(-1, 1))
# encoded_labels = oe.transform(relationship.reshape(-1, 1)).toarray()
# encoded_labels_df = pd.DataFrame(encoded_labels, columns=oe.categories_[0])
# testdata.drop('relationship',axis=1, inplace=True)
# testdata = testdata.join(encoded_labels_df)

# oe.fit(race.reshape(-1, 1))
# encoded_labels = oe.transform(race.reshape(-1, 1)).toarray()
# encoded_labels_df = pd.DataFrame(encoded_labels, columns=oe.categories_[0])
# testdata.drop('race',axis=1, inplace=True)
# testdata = testdata.join(encoded_labels_df)

# oe.fit(sex.reshape(-1, 1))
# encoded_labels = oe.transform(sex.reshape(-1, 1)).toarray()
# encoded_labels_df = pd.DataFrame(encoded_labels, columns=oe.categories_[0])
# testdata.drop('sex',axis=1, inplace=True)
# testdata = testdata.join(encoded_labels_df)

# oe.fit(native_country.reshape(-1, 1))
# encoded_labels = oe.transform(native_country.reshape(-1, 1)).toarray()
# encoded_labels_df = pd.DataFrame(encoded_labels, columns=oe.categories_[0])
# testdata.drop('native.country',axis=1, inplace=True)
# testdata = testdata.join(encoded_labels_df)

# testdata = np.array(testdata)

# DT = DecisionTreeClassifier(min_samples_leaf = 13, min_samples_split = 28, max_depth = 9, random_state = 66)
# DT.fit(train, train_label)
# pred = DT.predict(testdata)
# print(pred)

# # write in txt
# f=open("predict\decisionTree.txt","w")
 
# for line in pred:
#     f.write(str(line)+'\n')
# f.close()


