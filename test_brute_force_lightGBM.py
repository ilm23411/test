
import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz 

X = pd.DataFrame([0,0,1,1,1,1,2,3,1,1])

#y = np.array([0,0,0,0,0,1,1,1,1,1])
y = np.array([0,1,0,1,1,1,0,0,0,0])

from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=1)


clf = clf.fit(X, y)

clf.tree_.children_left #array of left children
clf.tree_.children_right #array of right children
clf.tree_.feature #array of nodes splitting feature
clf.tree_.threshold #array of nodes splitting points # first value is the split value
clf.tree_.value #array of nodes values



#############
plt.scatter(X, y,)

##########
dot_data = tree.export_graphviz(clf, out_file=None, 
                      #feature_names=iris.feature_names,  
                      #class_names=iris.target_names,  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 