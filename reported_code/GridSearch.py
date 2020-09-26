import u_nn 
#import u_neural_net
import numpy as np
from itertools import product
from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
#from u_nn import get_model


def grid_search(X_train, Y_train):
    # Parameters
    input_shape = (256, 256, 1)

    start_neurons = [32,64]
    optimizer = ['Adam']
    dropout = [0.5, 0.35, 0.2]
    init = ['glorot_uniform']
    learn_rate = [0.0001, 0.00001]

    # K-Fold CV grid search
    combo = product(start_neurons, optimizer, dropout, init, learn_rate)
    print(combo)

    kf = KFold(n_splits=5, shuffle=False)
    scores = np.zeros((72, 3))  # score matrix for average of all folds
    for i, parameters in enumerate(combo):
        score = []
        for train_index, test_index in kf.split(X_train):
            #print(parameters[0], parameters[1], parameters[2], parameters[3])
            #print("optimizer" , parameters[0])
            #print("dropout" , parameters[1])
            #print("init" , parameters[2])
            #print("learn rate" ,parameters[3])
            model = u_nn.get_model(input_shape, parameters[0], parameters[1], parameters[2], parameters[3], parameter[4])

            # Train the model (K-1 folds)
            file_path = "unn_grid.h5"
            checkpoint = ModelCheckpoint(file_path , monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
            redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
            callbacks_list = [checkpoint, early, redonplat]
            model.fit(X_train[train_index], Y_train[train_index], epochs=1000, batch_size=4, verbose=2,
                      callbacks=callbacks_list, validation_split=0.1)
            model.load_weights(file_path)

            # Predict on Kth fold
            train_predictions = model.predict(X_train[test_index])
            train_predictions = np.argmax(train_predictions, axis=-1)

            # Scores
            (confusion_matrix, overall_precision, per_class_precision, loU) = u_nn.compute_metrics(Y_train[test_index], train_predictions)
            score.append([overall_precision, per_class_precision, loU])
        score = np.array(score) 
        scores[i, :] = np.mean(score, axis=0)
        print("scores=", scores,"parameters=", parameters)
        scores_table = "score_table.h5"
        np.save(scores_table, scores)

    return scores


X_train_rotated = np.load('../data/x_train_new.npy').reshape((-1, 256, 256, 1))
Y_train_rotated = np.load('../data/y_train_new.npy').reshape((-1, 256, 256, 1))
scores = grid_search(X_train_rotated, Y_train_rotated)
print(scores)
