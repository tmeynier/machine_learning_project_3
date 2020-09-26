import numpy as np
from keras import optimizers, losses, activations, models
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Conv2D, concatenate, Input, Dropout, MaxPooling2D, Conv2DTranspose, LeakyReLU
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
import keras
import tensorflow as tf
import u_nn

filename = "unn.h5"
nb_images_test = 10
start = 50
end = start + nb_images_test

X_test = u_nn.load_data('../data/test_images/', 50, end)
Y_test = u_nn.load_data('../data/test_labels/', 50, end)
X_test_rot = u_nn.load_data('../data/test_images_randomly_rotated/', 50, end)
Y_test_rot = u_nn.load_data('../data/test_labels_randomly_rotated/', 50, end)
Y_test_rot[Y_test_rot==3]=1
#load model
model = load_model(filename)

# summarize model.
model.summary()

test_predictions_normal = model.predict(X_test)
test_predictions_normal = np.argmax(test_predictions_normal, axis=-1)

test_predictions_rot = model.predict(X_test_rot)
test_predictions_rot = np.argmax(test_predictions_rot, axis=-1)


# Compute metrics
(confusion_matrix_n, overall_precision_n, per_class_precision_n, IoU_n) =u_nn.compute_metrics(Y_test, test_predictions_normal)
(confusion_matrix_r, overall_precision_r, per_class_precision_r, IoU_r) = u_nn.compute_metrics(Y_test_rot, test_predictions_rot)
print("Scores for the unrotated data : " )
print("overall_precision : %s "% overall_precision_n)
print("per_class_precision : %s "% per_class_precision_n)
print("IoU: %s "% IoU_n)

print("Scores for the rotated data  : ")
print("overall_precision : %s "% overall_precision_r)
print("per_class_precision : %s "% per_class_precision_r)
print("IoU: %s "% IoU_r)
