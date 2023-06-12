import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

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

print(traindata['Local-gov'])

trainlabel = np.genfromtxt("data\\trainlabel.txt",dtype=int)
trainlabel = np.array(trainlabel)


# # 方差降维
# var = traindata.var()
# numeric = traindata.columns
# variable = []
# for i in range(0,len(var)):
#     if var[i]>=0.1:   # 将阈值设置为0.1
#        variable.append(numeric[i+1])
# lowTrainData = traindata[variable]

# mean_corr = lowTrainData.corr()
# f, ax = plt.subplots(figsize=(14,8))
# ax = sns.heatmap(mean_corr,annot=False,cmap="YlGnBu", center=0.2)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
# plt.show()
# figure = ax.get_figure()
# figure.savefig('heatmap.jpg')

# lowTrainDataArr = np.array(lowTrainData)
# plt.figure(figsize=[15,15])
# ax = sns.heatmap(lowTrainDataArr, xticklabels=variable, yticklabels=variable)
# ax.set_title('Heatmap')  # 图标题
# plt.show()

# label_DF = pd.DataFrame(trainlabel)
# label_DF.rename(columns={0:'label'},inplace=True)


# fig,ax = plt.subplots(figsize=(7,7), dpi=80)
# ax.set_title("label distrubtion")
# sns.countplot(x = label_DF['label'])
# for p in ax.patches:
#     ax.annotate(f'\n{p.get_height()}', (p.get_x(), p.get_height()+50), color='black', size=15)
# plt.show()


# model = RandomForestRegressor(random_state=1, max_depth=10)
# traindata=pd.get_dummies(traindata)
# model.fit(traindata,trainlabel)

# features = traindata.columns
# importances = model.feature_importances_
# indices = np.argsort(importances[0:15]) 

# plt.figure(figsize=[12,12])
# plt.title('Feature Importances')
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Relative Importance')
# plt.show()


