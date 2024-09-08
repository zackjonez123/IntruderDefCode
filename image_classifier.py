import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def main():

#prepare data
    input_dir = '/home/pi/code/captured-images'
    categories = [(0, 'friendly'), (1, 'hostile')]

    data = []
    labels = []

    for category_idx, category in categories: 
        for file in os.listdir(os.path.join(input_dir, category)):
            img_path = os.path.join(input_dir, category, file) #location of images
            img = cv2.imread(img_path) #reading image
            data.append(img.flatten()) #adding image as a unidimensional array to data array
            labels.append(category_idx)

    data = np.asarray(data)
    labels = np.asarray(labels)

#train/test split
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, shuffle=True, stratify=labels) 

#train classifier
    classifier = SVC()
    parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
    grid_search = GridSearchCV(classifier, parameters)
    grid_search.fit(x_train, y_train) # find coefficients, applied to x_train (images) that results in y_train (labels)

#test performance (look up confusion matrix for alternative, look up how to save module to file)
best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(x_test) # future image results with training coefficients applied
score = accuracy_score(y_prediction, y_test)
print('{}% of samples were correctly classified')

if __name__ == '__main__':
    main()
