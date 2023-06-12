# Project 3 - Adult Census Income

## 环境

python：3.8.16

numpy：1.24.3

pandas: 2.0.2

matplotlib:3.7.1

seaborn:0.12.2

scikit-learn-1.2.2

scipy:1.10.1

lightgbm:3.3.5



## 代码

### 功能：

**demensionReduce.py:**  读入数据并做缺失值和降维处理。

**dicisionTreeReduce.py:** 通过sklearn训练得到决策树模型。

**lightgbmTest.py:** 通过lightgbm训练得到的决策树模型

### 运行代码：

由于通过lightgbm得到的决策树模型效果最好，所以如果要得到与 **testlabel.txt** 中相同的预测结果，只需要运行 **lightgbm.py** 即可，运行步骤如下：

1) 在与**lightgbm.py**同文件夹下创建 **predict** 文件夹， 并在该文件夹下创建**testlabel.txt** 文件，将*testdata.csv* 和 *traindata.cv*、*trainlabel.txt* 放在**data**文件夹下。文件位置如图：

   ![image-20230608171833521](C:\Users\Sakura Yang\AppData\Roaming\Typora\typora-user-images\image-20230608171833521.png)



2) 直接运行**lightgbm.py**, 得到的预测结果保存在**testlabel.txt** 文件中。

*注：**lightgbm.py** 和 **dicisionTreeReduce.py**中注释掉的代码均为调参时使用，预测结果时不需要！*

