from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import math





df = pd.read_csv('E388_2.csv',header=None) #读取经过特征筛选后的文件
y = df[100] #第100列为标签
X = df[range(100)] #0-99为特征
#print(y)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=4) #7:3划分数据集
#print(X_test)
print(y_test)
labelArr = y_test.values.tolist() #y_test转化为列表
print(labelArr)
grid = GridSearchCV(SVC(), param_grid={'C': [0.1, 1, 10,100,1000], 'gamma': [100, 10, 1, 0.1, 0.01]}, cv=5)
grid.fit(X_train, y_train) #网格搜索寻找最优参数
print("The best parameters are %s with a score of %0.2f"% (grid.best_params_, grid.best_score_))


clf = SVC(kernel="rbf", gamma=0.1, C=1, probability=True)
clf.fit(X_train,y_train)
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))
predictArr = clf.predict(X_test)
print(predictArr)

def performance(labelArr, predictArr):
    #定义评估函数
    TP = 0.; TN = 0.; FP = 0.; FN = 0.
    for i in range(len(labelArr)):
        if labelArr[i] == 1 and predictArr[i] == 1:
            TP += 1.
        if labelArr[i] == 1 and predictArr[i] == 0:
            FN += 1.
        if labelArr[i] == 0 and predictArr[i] == 1:
            FP += 1.
        if labelArr[i] == 0 and predictArr[i] == 0:
            TN += 1.
    if (TP + FN)==0:
        SN=0
    else:
        SN = TP/(TP + FN) #Sensitivity = TP/P  and P = TP + FN
    if (FP+TN)==0:
        SP=0
    else:
        SP = TN/(FP + TN) #Specificity = TN/N  and N = TN + FP
    if (TP+FP)==0:
        precision=0
    else:
        precision=TP/(TP+FP)
    if (TP+FN)==0:
        recall=0
    else:
        recall=TP/(TP+FN)
    GM=math.sqrt(recall*SP)

    return SN,SP,precision


confusion_matrix = confusion_matrix(predictArr,y_test)
print(performance(labelArr,predictArr))
print(confusion_matrix)
f,ax = plt.subplots()
sns.heatmap(confusion_matrix,annot = True,ax = ax,)
ax.set_title('E.coli')
ax.set_xlabel('predict label')
ax.set_ylabel('true label')
plt.show()


