
# importing necessary libraries
import argparse
import os
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import mlflow
import mlflow.sklearn

# create an argument parser to take input arguments from command line
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument('--criterion', type=str, default='gini',
                        help='The function to measure the quality of a split')
    parser.add_argument('--max-depth', type=int, default=None,
                        help='The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples.')
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--registered_model_name", type=str, help="model name")

    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    # enable autologging
    mlflow.sklearn.autolog()

    # print input arguments
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    # load input data
    print("input data:", args.data)
    df = pd.read_csv(args.data)

    # log input hyperparameters

    mlflow.log_param('Criterion', str(args.criterion))
    mlflow.log_param('Max depth', str(args.max_depth))

    # split the data into training and testing sets
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_train_ratio,
    )


    # training a decision tree classifier


    # Extracting the label column
    y_train = train_df.pop("class")

    # convert the dataframe values to array
    X_train = train_df.values

    # Extracting the label column
    y_test = test_df.pop("class")

    # convert the dataframe values to array
    X_test = test_df.values

    # initialize and train a decision tree classifier
    tree_model = DecisionTreeClassifier(criterion=args.criterion, max_depth=args.max_depth)
    tree_model = tree_model.fit(X_train, y_train)
    tree_predictions = tree_model.predict(X_test)

    # compute and log model accuracy
    accuracy = tree_model.score(X_test, y_test)
    print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(accuracy))
    mlflow.log_metric('Accuracy', float(accuracy))

    # creating a confusion matrix
    cm = confusion_matrix(y_test, tree_predictions)
    print(cm)

    # set the name for the registered model
    registered_model_name="pima_decisiontree_model"

    ##########################
    #<save and register model>
    ##########################

    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=tree_model,
        registered_model_name=registered_model_name,
        artifact_path=registered_model_name
    )

    # # Saving the model to a file
    print("Saving the model via MLFlow")
    mlflow.sklearn.save_model(
        sk_model=tree_model,
        path=os.path.join(registered_model_name, "trained_model"),
    )
    ###########################
    #</save and register model>
    ###########################
   
    # end MLflow tracking
    mlflow.end_run()

if __name__ == '__main__':
    main()

