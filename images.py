# Standard scientific python imports

import matplotlib.pyplot as plt

# Datasets

from sklearn import datasets, svm, metrics

# Digits

digits = datasets.load_digits()
iris = datasets.load_iris()

# Let's look at the 10 images store in the images
# attribute of the dataset. Target gives us which image
# it represents

images_and_labels = list(zip(digits.images, digits.target))

plt.figure(figsize=(6*5, 4*5))

for index, (image, label) in enumerate(images_and_labels[:10]):
    plt.subplot(2, 10, index + 1)
    plt.axis('off')
    plt.imshow(image)
    plt.title('Training: %i' % label)

# Create a classifier

n_samples = len(digits.images)
data = digits.images.reshape(n_samples, -1)
classifier = svm.SVC(gamma=0.001)

# We learn using the first half of the digits

classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

# We forecast using the second half of the digits

expected = digits.target[n_samples / 2:]
predicted = classifier.predict(data[n_samples / 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

# Let's look at the prediction now

images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))

for index, (image, prediction) in enumerate(images_and_predictions[:10]):
    plt.subplot(2, 10, index + 11)
    plt.axis('off')
    plt.imshow(image)
    plt.title('Prediction: %i' % prediction)

plt.show()
