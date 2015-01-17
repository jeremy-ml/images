# Standard scientific python imports

import matplotlib.pyplot as plt

# Datasets

from sklearn import datasets

# Digits

digits = datasets.load_digits()
iris = datasets.load_iris()

# Draw
# Iterate over two list

images_and_labels = list(zip(digits.images, digits.target))

plt.figure(figsize=(6*3.13, 4*3.13))

for index, (image, label) in enumerate(images_and_labels[:10]):
    plt.subplot(2, 10, index + 1)
    plt.axis('off')
    plt.imshow(image)
    plt.title('Training: %i' % label)

plt.show()
