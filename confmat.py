"""
********** Confusion Matrix ********
"""
from unittest import TestLoader
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

y_pred = []
y_true = []

# Iterate through test data
for inputs, labels in testloader:


# categories
classes = ('Friendly', 'Hostile')

# build confusion matrix
