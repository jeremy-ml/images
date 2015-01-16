# Standard scientific python imports

import matplotlib.pyplot as plt

# Datasets

from sklearn import datasets

# Digits

digits = datasets.load_digits()
iris = datasets.load_iris()

# Draw

images_and_labels = list(zip(digits.images, digits.target))

for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image)
    plt.title('Training: %i' % label)

plt.show()
