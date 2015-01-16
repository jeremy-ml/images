# Standard scientific python imports

import matplotlib.pyplot as plt

# Datasets

from sklearn import datasets

# Digits

digits = datasets.load_digits()

# Draw

plt.imshow(digits.images[0])
plt.show()
