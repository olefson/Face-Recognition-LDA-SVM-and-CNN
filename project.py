from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Get the dataset

faces = fetch_olivetti_faces()

_, img_height, img_width = faces.images.shape

print(faces.images.shape)

# Split the dataset

N_IDENTITIES = len(np.unique(faces.target)) # how many different individuals are in the dataset
GALLERY_SIZE = 8                            # use the first GALLERY_SIZE images per individual for training, the rest for testing

gallery_indices = []
probe_indices = []
for i in range(N_IDENTITIES):
    indices = list(np.where(faces.target == i)[0])
    gallery_indices += indices[:GALLERY_SIZE]
    probe_indices += indices[GALLERY_SIZE:]

x_train = faces.images[gallery_indices].reshape(-1, img_height*img_width) # vectorize train images
y_train = faces.target[gallery_indices]
x_test = faces.images[probe_indices].reshape(-1, img_height*img_width)    # vectorize test images
y_test = faces.target[probe_indices]

print(x_train.shape, x_test.shape)

# Visualize image sets
def show_images(imgs, num_rows, num_cols):
    assert len(imgs) == num_rows*num_cols

    full = None
    for i in range(num_rows):
        row = None
        for j in range(num_cols):
            if row is None:
                row = imgs[i*num_cols+j].reshape(img_height, img_width)*255.0
            else:
                row = np.concatenate((row, imgs[i*num_cols+j].reshape(img_height, img_width)*255.0), axis=1)
        if full is None:
            full = row
        else:
            full = np.concatenate((full, row), axis=0)

    f = plt.figure(figsize=(num_cols, num_rows))
    plt.imshow(full, cmap='gray')
    plt.axis('off')
    plt.show()

print('TRAINING')
show_images(x_train, N_IDENTITIES, GALLERY_SIZE)
print('TESTING')
show_images(x_test, N_IDENTITIES, 10 - GALLERY_SIZE)

# LDA and SVM
# PCA Implementation using SVD
def pca(X, n_components):
    # standardize data
    X = StandardScaler().fit_transform(X)
    # compute SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # select n_components
    X_pca = U[:, :n_components].dot(np.diag(S[:n_components]))
    components = Vt[:n_components, :]
    return X_pca, components

# Apply PCA to dataset
n_components = 100
x_train_pca, components = pca(x_train, n_components)
x_test_pca = StandardScaler().fit_transform(x_test).dot(components.T)

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(x_train_pca, y_train)
y_pred_lda = lda.predict(x_test_pca)

# SVM
svm = SVC(kernel='linear')
svm.fit(x_train_pca, y_train)
y_pred_svm = svm.predict(x_test_pca)

# Evaluate
f1_lda = f1_score(y_test, y_pred_lda, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
print(f"F1 score LDA: {f1_lda}")
print(f"F1 score SVM: {f1_svm}")

# Confusion Matrix / Visualization
cm_lda = confusion_matrix(y_test, y_pred_lda)
cm_svm = confusion_matrix(y_test, y_pred_svm)

# visualize confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ConfusionMatrixDisplay(cm_lda).plot(ax=ax[0])
ax[0].set_title('LDA Confusion Matrix')
ax[0].tick_params(axis='x', rotation=90, labelsize=10)  # rotate x vals for clarity
ax[0].tick_params(axis='y', rotation=0, labelsize=10)   
ConfusionMatrixDisplay(cm_svm).plot(ax=ax[1])
ax[1].set_title('SVM Confusion Matrix')
ax[1].tick_params(axis='x', rotation=90, labelsize=10)  # rotate x vals for clarity
ax[1].tick_params(axis='y', rotation=0, labelsize=10)  
plt.show()

# FROM PREVIOUS PROJECT
# # visualize 2D representation using LDA
# lda_2d = LinearDiscriminantAnalysis(n_components=2)
# x_train_2d = lda_2d.fit_transform(x_train, y_train)

# plt.figure(figsize=(14, 12))
# for i in range(N_IDENTITIES):
#     plt.scatter(x_train_2d[y_train == i, 0], x_train_2d[y_train == i, 1], label=f'Class {i}', alpha=0.6)
# plt.title("2D Representation of Faces Using LDA")
# plt.xlabel("LDA 1")
# plt.ylabel("LDA 2")
# plt.legend(loc="best")
# plt.show()


# CNN with LeNet-5
# CNN preparation
x_train_cnn = faces.images[gallery_indices].reshape(-1, img_height, img_width, 1) # reshape training images for cnn
x_test_cnn = faces.images[probe_indices].reshape(-1, img_height, img_width, 1)   # reshape test images for cnn
y_train_cnn = faces.target[gallery_indices] #set training labels
y_test_cnn = faces.target[probe_indices]   #set test labels

# Normalize data
# converts pixel vals from int to float and scales to [0,1] by dividing by 255
x_train_cnn = x_train_cnn.astype(np.float32)
x_train_cnn /= 255

x_test_cnn = x_test_cnn.astype(np.float32)
x_test_cnn /= 255

# Split training data into training and validation sets
shuffle = np.random.permutation(len(x_train_cnn))
validation_size = int(0.2 * len(x_train_cnn)) # calc size of validation set

# validation set creation
x_val_cnn = x_train_cnn[shuffle[:validation_size]]
x_train_cnn = x_train_cnn[shuffle[validation_size:]]
y_val_cnn = y_train_cnn[shuffle[:validation_size]]
y_train_cnn = y_train_cnn[shuffle[validation_size:]]

# Define network architecture (LeNet-5)
model = tf.keras.models.Sequential() # sequential model
model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(img_height, img_width, 1))) #Conv2d layer (6 filters of size 5x5)
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2))) #AveragePooling2D layer (pool size 2x2)
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu')) #Conv2d layer (16 filters of size 5x5)
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2))) #AveragePooling2D layer (pool size 2x2)
model.add(tf.keras.layers.Flatten()) # flatten output of prev layer to 1D
model.add(tf.keras.layers.Dense(units=120, activation='relu')) #Dense layer (120 units)
model.add(tf.keras.layers.Dense(units=84, activation='relu')) #Dense layer (84 units)
model.add(tf.keras.layers.Dense(units=N_IDENTITIES, activation = 'softmax')) #Dense layer (N_IDENTITIES units)

# PRINT MODEL SUMMARY FOR DEBUGGING
print(model.summary())

# Training configuration
learning_rate = 0.0001 # set learning rate for optimizer
num_epochs = 20 # set number of epochs (full passes through dataset)
batch_size = 32 # set batch size for training
loss_function = 'sparse_categorical_crossentropy' # set loss function for training
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # set optimizer for training
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy']) # compile model

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=3, verbose=1) # config early stopping to monitor validation accuracy
model_save = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.keras', monitor='val_accuracy', save_best_only=True) # config model checkpoint to save best model

# Train
history = model.fit(x=x_train_cnn, y=y_train_cnn, epochs=num_epochs, batch_size=batch_size, validation_data=(x_val_cnn, y_val_cnn), callbacks=[early_stop, model_save])

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

# Results
# Restore weights from the best training point
model.load_weights('best_model.keras')

scores_train = model.evaluate(x_train_cnn, y_train_cnn, verbose=0)
scores_val = model.evaluate(x_val_cnn, y_val_cnn, verbose=0)
scores_test = model.evaluate(x_test_cnn, y_test_cnn, verbose=0)
print(f"TRAINING SET\nLoss: {scores_train[0]}, Accuracy: {scores_train[1]}")
print(f"VALIDATION SET\nLoss: {scores_val[0]}, Accuracy: {scores_val[1]}")
print(f"TEST SET\nLoss: {scores_test[0]}, Accuracy: {scores_test[1]}")

# Confusion Matrix for CNN
y_pred_cnn = model.predict(x_test_cnn)
y_pred_cnn = np.argmax(y_pred_cnn, axis=1)
cm_cnn = confusion_matrix(y_test_cnn, y_pred_cnn)

# visualize confusion matrix
fig, ax = plt.subplots(figsize=(10, 7))
ConfusionMatrixDisplay(cm_cnn).plot(ax=ax)
ax.set_title('CNN Confusion Matrix')
ax.tick_params(axis='x', rotation=90, labelsize=10)
ax.tick_params(axis='y', rotation=0, labelsize=10)
plt.show()
