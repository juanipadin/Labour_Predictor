import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

person=     {'finance':    [4,5,6,7,5,4,6,5,4,3,3,6,6,4,2,4,6,7,3,3,5,7],
             'management': [1,2,3,4,5,6,7,7,5,4,5,6,4,5,6,4,3,3,4,3,3,3],
             'logistics':  [1,2,4,5,5,6,7,5,5,6,6,5,4,4,5,6,7,4,4,5,5,5],
             'get_work':   [1,0,1,0,1,0,1,0,0,1,1,1,1,0,0,0,1,0,1,0,1,0]
             }

Data=pd.DataFrame(person,columns=['finance','management','logistics','get_work'])

X=Data[['finance','management','logistics']]
y=Data['get_work']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

lr=LogisticRegression()
lr.fit(X_train,y_train)
y_prediction=lr.predict(X_test)

conf_mat=pd.crosstab(y_test,y_prediction,rownames=['True'],colnames=['prevision'])
sb.heatmap(conf_mat,annot=True)

print('Accurance: ', metrics.accuracy_score(y_test,y_prediction))

plt.show()