import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import time
import pickle

start = time.time()     # Set timer

### 1. Prepare data

input_dir = './Image_Classifier/Input'      # Set input directory
categories = ['empty', 'not_empty']         # Set input folders

data = []       # List to append the data to model
labels = []     # List to append data labels

# Iterate over the documents and read the data as np arrays
for cat_idx, category in enumerate(categories):     # Iterate over the labels directories
    for file in os.listdir(os.path.join(input_dir, category)):      # Iterate over the images
        img_path = os.path.join(input_dir, category, file)          # Save the images
        img = imread(img_path)                                      # Read the image using skimage
        img = resize(img, (15, 15))                                 # Resize the images to be 15X15
        data.append(img.flatten())                                  # Flatten the image and send it to the list
        labels.append(cat_idx)                                      # Store category label

# Transform data list to arrays
data = np.asarray(data)
label = np.asarray(labels)

# Measure time
end = time.time()
elapsed = end-start
print(f'Data read in {elapsed} seconds')

### 2. train/test split

# Shuffle: Mix up the data to assure randomness on the data (Avoid bias)
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=True, stratify=labels)

### 3. Train classifier

start = time.time()     # Set timer
# Define classification object instance
classifier = SVC()
# Define list of hyperparameters
parameters = [
    {
    'gamma':[0.01, 0.001, 0.0001], 
    'C':[1, 10, 100, 1000]}]
# Instanciate gridsearch
grid_search = GridSearchCV(classifier, parameters)
# Fit gridsearch
grid_search.fit(X_train, y_train)

# Measure time
end = time.time()
elapsed = end-start
print(f'Data read in {elapsed} seconds')

### 4. Test performance

# Get best estimator
best_estimator = grid_search.best_estimator_

# Predict on test data
y_pred = best_estimator.predict(X_test)

# Evaluate model
acc_score = accuracy_score(y_pred, y_test)
f1_sc = f1_score(y_pred, y_test)

# Print results
print(f'The accuracy of the classifier is {round(acc_score*100,2)}')
print(f'The f1 score of the classifier is {round(f1_sc*100,2)}')

### 5. Save the model
with open('./model.p', 'wb') as wf:
    pickle.dump(best_estimator, wf)