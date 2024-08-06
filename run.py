# Standard scientific Python imports
import matplotlib.pyplot as plt

import matplotlib.cbook as cbook
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

with cbook.get_sample_data('/home/rafdeajuda/projetos-python/modelo-k-nn-numbers/tres.png') as image_file:
    digits_images = plt.imread(image_file)

# convert image to NumPy array
digits_images = np.array(digits_images)

digits = datasets.load_digits()
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
# plt.show()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape(n_samples, -1)

# create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# split data into 50% train and 50% test subsets
# X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

X_train = digits.images[0][0:8]
X_test = digits_images[0]
y_train = digits.target[0:8]
y_test =  digits.target[:4]

print(X_train, X_test, y_train, y_test)

# learn the digits on the train subset
clf.fit(X_train, y_train)

# predict the value of the digit on the test subset
predicted = clf.predict(X_test)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f'Prediction: {prediction}')

print(
    f'Classionfication report for classifier {clf}:\n'
    f'{metrics.classification_report(y_test, predicted)}\n'
)