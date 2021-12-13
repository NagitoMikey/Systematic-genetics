from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc


df3 = pd.read_csv('E388_2.csv',header=None)
y3= df3[100]
X3= df3[range(100)]
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3,y3,test_size=0.3,random_state=4)
clf3 = SVC(kernel="rbf", gamma=0.1, C=1, probability=True)
clf3.fit(X_train3,y_train3)
predictions3 = clf3.predict_proba(X_test3)# 获得测试集上训练模型得到的结果
false_positive_rate3, recall3, thresholds3 = roc_curve(y_test3, predictions3[:
, 1])# 获得真假阳性率
roc_auc3=auc(false_positive_rate3,recall3)


df2 = pd.read_csv('D1769_2.csv',header=None)
y2 = df2[100]
X2 = df2[range(100)]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2,test_size=0.3,random_state=4)
#clf2 = SVC(kernel="rbf", gamma=0.1, C=0.1, probability=True)
#clf2.fit(X_train2,y_train2)
predictions2 = clf3.predict_proba(X_test2)# 获得测试集上训练模型得到的结果
false_positive_rate2, recall2, thresholds2 = roc_curve(y_test2, predictions2[:
, 1])# 获得真假阳性率
roc_auc2=auc(false_positive_rate2,recall2)


df1 = pd.read_csv('A1978_2.csv',header=None)
y1 = df1[100]
X1 = df1[range(100)]
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y1,test_size=0.3,random_state=4)
#clf = SVC(kernel="rbf", gamma=0.01, C=100, probability=True)
#clf.fit(X_train1,y_train1)
predictions1 = clf3.predict_proba(X_test1)# 获得测试集上训练模型得到的结果
false_positive_rate1, recall1, thresholds1 = roc_curve(y_test1, predictions1[:
, 1])# 获得真假阳性率
roc_auc1=auc(false_positive_rate1,recall1)


plt.title('ROC')
plt.plot(false_positive_rate1, recall1, 'b', label='A.thaliana = %0.2f' % roc_auc1)
plt.plot(false_positive_rate2, recall2, 'r', label='D.melanogaster = %0.2f' % roc_auc2)
plt.plot(false_positive_rate3, recall3, 'y', label='E.coli = %0.2f' % roc_auc3)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()