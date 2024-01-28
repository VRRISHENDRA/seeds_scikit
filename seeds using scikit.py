Python 3.10.2 (tags/v3.10.2:a58ebcc, Jan 17 2022, 14:12:15) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import pandas as pd
tot=pd.read_csv(r"F:\archive\Seed_Data.csv")
tot.describe
<bound method NDFrame.describe of          A      P       C     LK     WK  A_Coef    LKG  target
0    15.26  14.84  0.8710  5.763  3.312   2.221  5.220       0
1    14.88  14.57  0.8811  5.554  3.333   1.018  4.956       0
2    14.29  14.09  0.9050  5.291  3.337   2.699  4.825       0
3    13.84  13.94  0.8955  5.324  3.379   2.259  4.805       0
4    16.14  14.99  0.9034  5.658  3.562   1.355  5.175       0
..     ...    ...     ...    ...    ...     ...    ...     ...
205  12.19  13.20  0.8783  5.137  2.981   3.631  4.870       2
206  11.23  12.88  0.8511  5.140  2.795   4.325  5.003       2
207  13.20  13.66  0.8883  5.236  3.232   8.315  5.056       2
208  11.84  13.21  0.8521  5.175  2.836   3.598  5.044       2
209  12.30  13.34  0.8684  5.243  2.974   5.637  5.063       2

[210 rows x 8 columns]>
X=tot.iloc[:,0:7]
X.info
<bound method DataFrame.info of          A      P       C     LK     WK  A_Coef    LKG
0    15.26  14.84  0.8710  5.763  3.312   2.221  5.220
1    14.88  14.57  0.8811  5.554  3.333   1.018  4.956
2    14.29  14.09  0.9050  5.291  3.337   2.699  4.825
3    13.84  13.94  0.8955  5.324  3.379   2.259  4.805
4    16.14  14.99  0.9034  5.658  3.562   1.355  5.175
..     ...    ...     ...    ...    ...     ...    ...
205  12.19  13.20  0.8783  5.137  2.981   3.631  4.870
206  11.23  12.88  0.8511  5.140  2.795   4.325  5.003
207  13.20  13.66  0.8883  5.236  3.232   8.315  5.056
208  11.84  13.21  0.8521  5.175  2.836   3.598  5.044
209  12.30  13.34  0.8684  5.243  2.974   5.637  5.063

[210 rows x 7 columns]>
#in the above tot.iloc we are not including last row
y=tot.iloc[:,7]
y.describe
<bound method NDFrame.describe of 0      0
1      0
2      0
3      0
4      0
      ..
205    2
206    2
207    2
208    2
209    2
Name: target, Length: 210, dtype: int64>
#in this we are only including last row
import sklearn
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=13)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
clf=svm.SVC()
clf.fit(X_train,y_train)
SVC()
pred_clf=clf.predict(X_test)
sklearn.metrics.accuracy_score(y_test,pred_clf)
0.9523809523809523
sklearn.metrics.classification_report(y_test,pred_clf)
'              precision    recall  f1-score   support\n\n           0       0.80      1.00      0.89         8\n           1       1.00      0.95      0.97        19\n           2       1.00      0.93      0.97        15\n\n    accuracy                           0.95        42\n   macro avg       0.93      0.96      0.94        42\nweighted avg       0.96      0.95      0.95        42\n'
print(sklearn.metrics.classification_report(y_test,pred_clf))
              precision    recall  f1-score   support

           0       0.80      1.00      0.89         8
           1       1.00      0.95      0.97        19
           2       1.00      0.93      0.97        15

    accuracy                           0.95        42
   macro avg       0.93      0.96      0.94        42
weighted avg       0.96      0.95      0.95        42

