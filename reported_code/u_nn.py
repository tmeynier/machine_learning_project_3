import numpy as np 
from keras import optimizers, losses, activations, models
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Conv2D, concatenate, Input, Dropout, MaxPooling2D, Conv2DTranspose, LeakyReLU
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix
import keras
import tensorflow as tf
#################################### TOOL FUNCTIONS ######################################

def load_data(filepath, start_index, end_index):

    data = []
    for i in range(start_index,end_index):
        filename = filepath + 'sample-' + str(i) + '.npy'
        image_3d = np.load(filename).reshape((-1, 256, 256, 1))

        if i == start_index:
            data = image_3d
        else:
            data = np.concatenate((data, image_3d), axis=0)
    
    return data

def build_model(input_layer, start_neurons, dropout, init):

    conv1 = Conv2D(start_neurons * 1, (3, 3), padding="same", kernel_initializer=init)(input_layer)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = Conv2D(start_neurons * 1, (3, 3), padding="same", kernel_initializer=init)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(dropout/2)(pool1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), padding="same", kernel_initializer=init)(pool1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = Conv2D(start_neurons * 2, (3, 3), padding="same", kernel_initializer=init)(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(dropout)(pool2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), padding="same", kernel_initializer=init)(pool2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = Conv2D(start_neurons * 4, (3, 3), padding="same", kernel_initializer=init)(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), padding="same", kernel_initializer=init)(pool3)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = Conv2D(start_neurons * 8, (3, 3), padding="same", kernel_initializer=init)(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), padding="same", kernel_initializer=init)(pool4)
    convm = LeakyReLU(alpha=0.1)(convm)
    convm = Conv2D(start_neurons * 16, (3, 3), padding="same", kernel_initializer=init)(convm)
    convm = LeakyReLU(alpha=0.1)(convm)
    
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(convm)
    deconv4 = LeakyReLU(alpha=0.1)(deconv4)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), padding="same", kernel_initializer=init)(uconv4)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), padding="same", kernel_initializer=init)(uconv4)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(uconv4)
    deconv3 = LeakyReLU(alpha=0.1)(deconv3)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(dropout)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), padding="same", kernel_initializer=init)(uconv3)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), padding="same", kernel_initializer=init)(uconv3)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(uconv3)
    deconv2 = LeakyReLU(alpha=0.1)(deconv2)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(dropout)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), padding="same", kernel_initializer=init)(uconv2)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), padding="same", kernel_initializer=init)(uconv2)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(uconv2)
    deconv1 = LeakyReLU(alpha=0.1)(deconv1)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(dropout)(uconv1)
    uconv1 = Dropout(dropout)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), padding="same", kernel_initializer=init)(uconv1)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), padding="same", kernel_initializer=init)(uconv1)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    
    output_layer = Conv2D(3, (1,1), padding="same", activation="softmax")(uconv1)
       
    return output_layer

def get_model(input_shape, start_neurons, optimizer, dropout, init, learn_rate):

    input_layer = Input(shape=input_shape)
    predictions = build_model(input_layer, start_neurons, dropout, init)
    model = Model(inputs=input_layer, outputs=predictions)
    if(optimizer == 'Adam') :
        opt = optimizers.Adam(learning_rate=learn_rate)
    elif(optimizer == "SGD"):
        opt = optimizers.SGD(learning_rate=learn_rate)
    elif(optimizer == "Adamax"):
        opt = optimizers.Adamax(learning_rate=learn_rate)
    elif(optimizer == 'RMSprop'):
        opt = optimizers.RMSprop(learning_rate=learn_rate)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    
    return model


def build_model_3_layers(input_layer, start_neurons, dropout, init):

    conv1 = Conv2D(start_neurons * 1, (3, 3), padding="same", kernel_initializer=init)(input_layer)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = Conv2D(start_neurons * 1, (3, 3), padding="same", kernel_initializer=init)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(dropout/2)(pool1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), padding="same", kernel_initializer=init)(pool1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = Conv2D(start_neurons * 2, (3, 3), padding="same", kernel_initializer=init)(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(dropout)(pool2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), padding="same", kernel_initializer=init)(pool2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = Conv2D(start_neurons * 4, (3, 3), padding="same", kernel_initializer=init)(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)

    # Middle
    convm = Conv2D(start_neurons * 8, (3, 3), padding="same", kernel_initializer=init)(pool3)
    convm = LeakyReLU(alpha=0.1)(convm)
    convm = Conv2D(start_neurons * 8, (3, 3), padding="same", kernel_initializer=init)(convm)
    convm = LeakyReLU(alpha=0.1)(convm)
    
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(convm)
    deconv3 = LeakyReLU(alpha=0.1)(deconv3)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(dropout)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), padding="same", kernel_initializer=init)(uconv3)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), padding="same", kernel_initializer=init)(uconv3)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(uconv3)
    deconv2 = LeakyReLU(alpha=0.1)(deconv2)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(dropout)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), padding="same", kernel_initializer=init)(uconv2)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), padding="same", kernel_initializer=init)(uconv2)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(uconv2)
    deconv1 = LeakyReLU(alpha=0.1)(deconv1)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(dropout)(uconv1)
    uconv1 = Dropout(dropout)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), padding="same", kernel_initializer=init)(uconv1)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), padding="same", kernel_initializer=init)(uconv1)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    
    output_layer = Conv2D(3, (1,1), padding="same", activation="softmax")(uconv1)
       
    return output_layer


def get_model_3_layers(input_shape, start_neurons, optimizer, dropout, init, learn_rate):

    input_layer = Input(shape=input_shape)
    predictions = build_model_3_layers(input_layer, start_neurons, dropout, init)
    model = Model(inputs=input_layer, outputs=predictions)
    if(optimizer == 'Adam') :
        opt = optimizers.Adam(learning_rate=learn_rate)
    elif(optimizer == "SGD"):
        opt = optimizers.SGD(learning_rate=learn_rate)
    elif(optimizer == "Adamax"):
        opt = optimizers.Adamax(learning_rate=learn_rate)
    elif(optimizer == 'RMSprop'):
        opt = optimizers.RMSprop(learning_rate=learn_rate)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    
    return model


def build_model_5_layers(input_layer, start_neurons, dropout, init):

    conv1 = Conv2D(start_neurons * 1, (3, 3), padding="same", kernel_initializer=init)(input_layer)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = Conv2D(start_neurons * 1, (3, 3), padding="same", kernel_initializer=init)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(dropout/2)(pool1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), padding="same", kernel_initializer=init)(pool1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = Conv2D(start_neurons * 2, (3, 3), padding="same", kernel_initializer=init)(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(dropout)(pool2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), padding="same", kernel_initializer=init)(pool2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = Conv2D(start_neurons * 4, (3, 3), padding="same", kernel_initializer=init)(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), padding="same", kernel_initializer=init)(pool3)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = Conv2D(start_neurons * 8, (3, 3), padding="same", kernel_initializer=init)(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout)(pool4)

    conv5 = Conv2D(start_neurons * 16, (3, 3), padding="same", kernel_initializer=init)(pool4)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = Conv2D(start_neurons * 16, (3, 3), padding="same", kernel_initializer=init)(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    pool5 = MaxPooling2D((2, 2))(conv5)
    pool5 = Dropout(dropout)(pool5)

    # Middle
    convm = Conv2D(start_neurons * 32, (3, 3), padding="same", kernel_initializer=init)(pool5)
    convm = LeakyReLU(alpha=0.1)(convm)
    convm = Conv2D(start_neurons * 32, (3, 3), padding="same", kernel_initializer=init)(convm)
    convm = LeakyReLU(alpha=0.1)(convm)

    deconv5 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(convm)
    deconv5 = LeakyReLU(alpha=0.1)(deconv5)
    uconv5 = concatenate([deconv5, conv5])
    uconv5 = Dropout(dropout)(uconv5)
    uconv5 = Conv2D(start_neurons * 16, (3, 3), padding="same", kernel_initializer=init)(uconv5)
    uconv5 = LeakyReLU(alpha=0.1)(uconv5)
    uconv5 = Conv2D(start_neurons * 16, (3, 3), padding="same", kernel_initializer=init)(uconv5)
    uconv5 = LeakyReLU(alpha=0.1)(uconv5)

    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(uconv5)
    deconv4 = LeakyReLU(alpha=0.1)(deconv4)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), padding="same", kernel_initializer=init)(uconv4)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), padding="same", kernel_initializer=init)(uconv4)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(uconv4)
    deconv3 = LeakyReLU(alpha=0.1)(deconv3)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(dropout)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), padding="same", kernel_initializer=init)(uconv3)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), padding="same", kernel_initializer=init)(uconv3)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(uconv3)
    deconv2 = LeakyReLU(alpha=0.1)(deconv2)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(dropout)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), padding="same", kernel_initializer=init)(uconv2)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), padding="same", kernel_initializer=init)(uconv2)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(uconv2)
    deconv1 = LeakyReLU(alpha=0.1)(deconv1)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(dropout)(uconv1)
    uconv1 = Dropout(dropout)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), padding="same", kernel_initializer=init)(uconv1)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), padding="same", kernel_initializer=init)(uconv1)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    
    output_layer = Conv2D(3, (1,1), padding="same", activation="softmax")(uconv1)
       
    return output_layer

def get_model_5_layers(input_shape, start_neurons, optimizer, dropout, init, learn_rate):

    input_layer = Input(shape=input_shape)
    predictions = build_model_5_layers(input_layer, start_neurons, dropout, init)
    model = Model(inputs=input_layer, outputs=predictions)
    if(optimizer == 'Adam') :
        opt = optimizers.Adam(learning_rate=learn_rate)
    elif(optimizer == "SGD"):
        opt = optimizers.SGD(learning_rate=learn_rate)
    elif(optimizer == "Adamax"):
        opt = optimizers.Adamax(learning_rate=learn_rate)
    elif(optimizer == 'RMSprop'):
        opt = optimizers.RMSprop(learning_rate=learn_rate)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    
    return model

def compute_metrics(Y_train, train_predictions):

    nb_samples = Y_train.shape[0]

    confusion_matrix = np.zeros((3,3))
    
    for s in range(nb_samples):
        current_sample_truth = Y_train[s,:,:,0]
        current_sample_hat = train_predictions[s,:,:]
        width, height = current_sample_truth.shape

        for i in range(width):
            for j in range(height):
                confusion_matrix[current_sample_truth[i,j],current_sample_hat[i,j]] += 1

    correct_class_0 = confusion_matrix[0,0]
    correct_class_1 = confusion_matrix[1,1]
    correct_class_2 = confusion_matrix[2,2]
    
    total_class_0 = np.sum(confusion_matrix[0,:])
    total_class_1 = np.sum(confusion_matrix[1,:])
    total_class_2 = np.sum(confusion_matrix[2,:])

    mean_class_0 = correct_class_0*1.0/total_class_0
    mean_class_1 = correct_class_1*1.0/total_class_1
    mean_class_2 = correct_class_2*1.0/total_class_2

    overall_precision = (correct_class_0 + correct_class_1 + correct_class_2)*1.0/(total_class_0 + total_class_1 + total_class_2)
    per_class_precision = (mean_class_0 + mean_class_1 + mean_class_2)/3.0

    LoU = 0 
    for i in range(len(confusion_matrix)):
        numerator = confusion_matrix[i,i]
        denominator = np.sum(confusion_matrix[i,:]) + np.sum(confusion_matrix[:,i]) - confusion_matrix[i,i]
        LoU += numerator/denominator

    loU = LoU/3.0

    return confusion_matrix, overall_precision, per_class_precision, loU

######################################### MAIN ####################################################

# Parameters
nb_images_train = 50
nb_images_test = 10
start_neurons = 64

if __name__ == '__main__':
    # Load the datasets
    X_train = load_data('../data/train_images/', 0, nb_images_train)
    Y_train = load_data('../data/train_labels/', 0, nb_images_train)
    X_train_rotated = np.load('../data/x_train_new.npy').reshape((-1, 256, 256, 1))
    Y_train_rotated = np.load('../data/y_train_new.npy').reshape((-1, 256, 256, 1))

    # Train the model 
    model = get_model((256, 256, 1), start_neurons, "Adam", 0.2, "glorot_uniform", 0.0001)
    #model.load_weights("unn.h5")
    file_path = "unn.h5"

    # Train the model 
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]  
    model.fit(X_train_rotated, Y_train_rotated, epochs=1000, batch_size=4, verbose=2, callbacks=callbacks_list, validation_split=0.1)
    model.load_weights(file_path)

    # Evaluate performance
    train_predictions = model.predict(X_train_rotated)
    train_predictions = np.argmax(train_predictions, axis=-1)

    # Compute metrics
    (confusion_matrix, overall_precision, per_class_precision, loU) = compute_metrics(Y_train_rotated, train_predictions)
    print("confusion_matrix")
    print(confusion_matrix)
    print("overall_precision")
    print(overall_precision)
    print("per_class_precision")
    print(per_class_precision)
    print("loU")
    print(loU)
