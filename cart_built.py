import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydotplus

df = pd.read_csv("heart.csv")
df.head()
df.describe()

feature_cols=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
X=df[feature_cols]
Y=df.target

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
clf = DecisionTreeClassifier(criterion="gini")
clf = clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)
dtree_cm_id3=confusion_matrix(Y_test,Y_pred)

print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('ID3_Built-In.png')
Image(graph.create_png())