import numpy as np
import pandas as pd
%matplotlib inline

import random
from pprint import pprint

from decision_tree_functions import decision_tree_algorithm, decision_tree_predictions
from helper_functions import train_test_split, calculate_accuracy

df = pd.read_csv("heart2.csv")
df=df.rename(columns={"target":"label"})

df.head()

random.seed(0)
train_df, test_df = train_test_split(df, test_size=0.1)

indices=[204,159,219,174,184,295,269,119,193,154,51,249,278,229,208,302,58,220,18,228,11,300,70,138,122,164,191,80,27,123,157]    

test_df=df.loc[indices]
train_df=df.drop(indices)

def bootstrapping(train_df, n_bootstrap):
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]
    
    return df_bootstrapped

def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features)
        forest.append(tree)
    
    return forest

def random_forest_predictions(test_df, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    print(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]
    print("And Random Forest Prediction:::")
    print(random_forest_predictions)
    
    return random_forest_predictions

forest = random_forest_algorithm(train_df, n_trees=7, n_bootstrap=120, n_features=7, dt_max_depth=4)
predictions = random_forest_predictions(test_df, forest)
accuracy = calculate_accuracy(predictions, test_df.label)

print("Accuracy = {}".format(accuracy))

