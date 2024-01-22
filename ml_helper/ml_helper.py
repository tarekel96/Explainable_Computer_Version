import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.spatial import distance
import numpy as np
import keras
import tensorflow as tf
import os
import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from skimage.feature import hog
from datetime import datetime
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

def today_date():
    return f'{datetime.today().month}.{datetime.today().day}.{datetime.today().year}'

def class_0_accuracy(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    total_samples_per_class = cm.sum(axis=1)
    accuracies_per_class = cm.diagonal() / total_samples_per_class
    class_0_accuracy = accuracies_per_class[0]
    return class_0_accuracy

def class_1_accuracy(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    total_samples_per_class = cm.sum(axis=1)
    accuracies_per_class = cm.diagonal() / total_samples_per_class
    class_1_accuracy = accuracies_per_class[1]
    return class_1_accuracy

def get_model_scores(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    total_samples_per_class = cm.sum(axis=1)
    accuracies_per_class = cm.diagonal() / total_samples_per_class

    class_0_accuracy = accuracies_per_class[0]
    class_1_accuracy = accuracies_per_class[1]

    metrics = (acc, prec, recall, f1, roc, class_0_accuracy, class_1_accuracy)
    return metrics

def create_explainable_hog(df_hog, img_scale, weights, threshold, index):
    threshold = np.percentile(weights, 90)
    high_weight_mask = weights > threshold
    filtered_weights = np.where(high_weight_mask, weights, 0).reshape(-1)
    
    ex_img = df_hog['hog_img'][index].flatten()

    for i in range(len(ex_img)):
        if filtered_weights[i] == 0:
            ex_img[i] = 0
    
    return ex_img.reshape(img_scale, img_scale)

class HOGTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, orientations=9, pixels_per_cell=(8,8),
                 cells_per_block=(2,2)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        hog_X = []
        img_scale = X[0].shape[1]

        for x in X:
            hog_fd, _ = hog(
                image=np.array(x).reshape((img_scale, img_scale, 3)),
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                visualize=True,
                channel_axis=2
            )
            hog_X.append(hog_fd)
        
        hog_X = np.array(hog_X)

        hog_wide_X = np.zeros((len(hog_X), hog_X[0].shape[0]))
        for indx in range(len(hog_X)):
            for feature in range(hog_X[0].shape[0]):
                hog_wide_X[indx][feature] = hog_X[indx][feature]
        
        return hog_wide_X
    
def hog_svm_hyperparameter_tester(img_df, param_grid, experiment_name, n_splits=3, seed=12172023):
    
    # preprocessed_img = PIL image converted to numpy image
    X, y = np.array(img_df['np_img'][:]), np.array(img_df['label'][:]).astype(int)

    log_file = './experiment_logs/{}_{}'.format(today_date(), experiment_name)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    random.seed(seed)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scorers = {
        'roc_auc': make_scorer(roc_auc_score),
        'class_0_accuracy': make_scorer(class_0_accuracy),
        'class_1_accuracy': make_scorer(class_1_accuracy),
        'precision': make_scorer(precision_score),
        'accuracy': make_scorer(accuracy_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }

    # pipeline
    pipeline = Pipeline([
        ('HOG', HOGTransformer()),
        ('SVM', SVC())
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorers,
        refit='roc_auc',
        cv=skf,
        verbose=2
    )

    grid_search.fit(X, y)

    cv_results = grid_search.cv_results_
    for i in range(len(cv_results['params'])):
        print('Parameters: {}'.format(cv_results['params'][i]))
        logging.info('Parameters: {}'.format(cv_results['params'][i]))
        for scorer in scorers:
            print('{}: {}'.format(scorer, cv_results[f'mean_test_{scorer}'][i]))
            logging.info('{}: {}'.format(scorer, cv_results[f'mean_test_{scorer}'][i]))
        print('\n'); logging.info('\n')
    
    print('Best parameters found: {}'.format(grid_search.best_params_))
    print('Best cross-validated ROC AUC Score: {}'.format(grid_search.best_score_))
    logging.info('Best parameters found: {}'.format(grid_search.best_params_))
    logging.info('Best cross-validated ROC AUC Score: {}'.format(grid_search.best_score_))

    try:
        file_handler.close()
    except Exception as e:
        print('Error shutting down the logging.\nError msg: {}'.format(e))
        pass

    return grid_search