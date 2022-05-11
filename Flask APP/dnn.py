import os
import pathlib
import numpy as np
import pandas as pd
from matplotlib import cm
from collections import Counter
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam

import logging
logging.getLogger('tensorflow').disabled = True

from variables import*
from util import*

class InventoryForecasting(object):
    def __init__(self):
        X, Y, Xtest, minmax_scaler = load_Data()
        self.X = X
        self.Y = Y
        self.Xtest = Xtest
        self.minmax_scaler = minmax_scaler

    def classifier(self):
        n_features = self.X.shape[1]
        inputs = Input(shape=(n_features,))
        x = Dense(dense1, activation='relu')(inputs)
        x = Dense(dense2, activation='relu')(x)
        x = Dense(dense2, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(keep_prob)(x)
        x = Dense(dense3, activation='relu')(x)
        x = Dense(dense3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(keep_prob)(x)
        x = Dense(dense4, activation='relu')(x)
        x = Dense(dense4, activation='relu')(x)
        outputs = Dense(dense_out, activation='sigmoid')(x)

        self.model = Model(
                        inputs,
                        outputs
                        )

    def train(self):
        self.model.compile(
                    loss='mse',
                    optimizer=Adam(learning_rate),
                    # metrics=['accuracy']
                        )

        self.history = self.model.fit(
                            self.X,
                            self.Y,
                            batch_size=batch_size,
                            epochs=num_epoches,
                            validation_split=validation_split
                            )
        self.plot_loss()
        self.save_model()

    def plot_loss(self):
        loss_train = self.history.history['loss']
        loss_val = self.history.history['val_loss']
        plt.plot(np.arange(1,num_epoches+1), loss_train, 'r', label='Training loss')
        plt.plot(np.arange(1,num_epoches+1), loss_val, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(loss_comparison_img)
        plt.legend()
        plt.show()

    def save_model(self):  # Saving the trained model
        print("Saving the model !!!")
        self.model.save(model_weights)

    def TFconverter(self): # For deployment in the mobile devices quantization of the model using tensorflow lite
        converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_weights)
        converter.target_spec.supported_ops = [
                                tf.lite.OpsSet.TFLITE_BUILTINS,   # Handling unsupported tensorflow Ops 
                                tf.lite.OpsSet.SELECT_TF_OPS 
                                ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]      # Set optimization default and it configure between latency, accuracy and model size
        tflite_model = converter.convert()

        model_converter_file = pathlib.Path(model_converter) 
        model_converter_file.write_bytes(tflite_model) # save the tflite model in byte format
 
    def TFinterpreter(self):
        self.interpreter = tf.lite.Interpreter(model_path=model_converter) # Load tflite model
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details() # Get input details of the model
        self.output_details = self.interpreter.get_output_details() # Get output details of the model

    def Inference(self, features):
        features = features.astype(np.float32)
        input_shape = self.input_details[0]['shape']
        assert np.array_equal(input_shape, features.shape), "Input tensor hasn't correct dimension"

        self.interpreter.set_tensor(self.input_details[0]['index'], features)

        self.interpreter.invoke() # set the inference

        output_data = self.interpreter.get_tensor(self.output_details[0]['index']) # Get predictions
        return output_data

    def Visualize_Predictions(self):
        Y = []
        P = []
        for y, x in zip(self.Y[:1000], self.X[:1000]):
            y = y.reshape(-1, 1)
            x = x.reshape(1, -1)

            y = int(self.minmax_scaler.inverse_transform(y).squeeze())
            p = int(self.minmax_scaler.inverse_transform(self.Inference(x)).squeeze())

            Y.append(y)
            P.append(p)
        
        Y = np.array(Y)
        P = np.array(P)
        e = np.abs(P-Y)

        plt.plot(e // 5)
        plt.title('Error Analysis')
        plt.xlabel('time')
        plt.ylabel('error')
        plt.savefig(error_analysis_img)
        plt.show()

        plt.plot(Y, 'r', label='Ground Truths')
        plt.plot(P, 'b', label='Predictions')
        plt.title('Model Results')
        plt.xlabel('time')
        plt.ylabel('sales')
        plt.savefig(model_results_img)  
        plt.show()

    def run(self):
        if not os.path.exists(model_converter):
            if not os.path.exists(model_weights):
                self.classifier()
                self.train()

            self.TFconverter()
        self.TFinterpreter()   
        # self.Visualize_Predictions()

model = InventoryForecasting()
model.run()