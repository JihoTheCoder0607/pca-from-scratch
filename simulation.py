import tensorflow as tf
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = np.array(X_train).flatten().reshape(60000, 28 * 28) / 255.0
y_train = np.array(y_train)
X_test = np.array(X_test).flatten().reshape(10000, 28 * 28) / 255.0
y_test = np.array(y_test)

from pca import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=8)

X_transformed = pca.fit_transform(X_train)
X_reconstructed = pca.reconstruct(X_transformed)

image = X_train[0].reshape(28, 28)
image_reconstructed = X_reconstructed[0].reshape(28, 28)

plt.imshow(image)
plt.show()

plt.imshow(image_reconstructed)
plt.show()