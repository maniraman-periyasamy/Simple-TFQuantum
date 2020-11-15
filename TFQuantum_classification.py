"""
This is a simple classification model to compare the performane of Quantum models
and hybrid Quantum/classical model. Single layer classical dense network is also
included to check the performance.

"""



import numpy as np
import sympy
import matplotlib.pyplot as plt
import os


# For Quantum ML
import cirq
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_quantum as tfq

from DataHandler import Data_Handler






class TFQuantum:
    """
    This class creates, trains and test all three models for the given data handler object (individual Dataset).
    """
    def __init__(self, dataHandler):

        self.dataHandler = dataHandler

    def createModel(self, model_circuit, model_readout,type = "Quantum"):

        if type == "Quantum":
            self.model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(), dtype=tf.string),
                tfq.layers.PQC(model_circuit, model_readout),
            ])

        elif type == "Hybrid":

            input = tf.keras.Input(shape=() ,dtype=tf.dtypes.string)
            quatum_output = tfq.layers.PQC(model_circuit, model_readout)(input)
            classifier_output = tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid)(quatum_output)
            self.model = tf.keras.Model(inputs=input, outputs=classifier_output)

        elif type == "Single_Dense":
            input = tf.keras.Input(shape=((1,2)))
            classifier_output = tf.keras.layers.Dense(5, activation=tf.keras.activations.sigmoid)(input)
            classifier_output = tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid)(classifier_output)
            self.model = tf.keras.Model(inputs=input, outputs=classifier_output)

        self.model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            metrics=["accuracy"])

    def train(self, method = "Quantum"):

        if method == "Single_Dense":
            train_Circuit, train_label = self.dataHandler.get_trainData()
            val_Circuit, val_label = self.dataHandler.get_trainData()
        else:
            train_Circuit, train_label = self.dataHandler.get_valCircuit()
            val_Circuit, val_label = self.dataHandler.get_valCircuit()

        callBack = EarlyStopping(patience=10, restore_best_weights=True)
        model_history = self.model.fit(
                train_Circuit, train_label,
                batch_size=200,
                epochs=20,
                verbose=1,
                validation_data=(val_Circuit, val_label), callbacks=[callBack])

    def predict(self, method ="Quantum"):

        if method == "Single_Dense" or method == "Ground_Truth":
            test_Circuit, test_label = self.dataHandler.get_testData()
        else:
            test_Circuit, test_label = self.dataHandler.get_testCircuit()

        if method != "Ground_Truth":
            results = self.model.predict(test_Circuit)
            self.y_pred = np.argmax(results, axis=1)
            evaluation_results = self.model.evaluate(test_Circuit, test_label)
            print("Type: {}, Loss: {}, Accuracy: {}".format(method, evaluation_results[0], evaluation_results[1]))

        else:
            self.y_pred = np.argmax(test_label, axis=1)

    def plot(self,axis,plotNu, method):
        i = int(plotNu%2)
        j = int(plotNu/2)

        test_Circuit, test_label = self.dataHandler.get_testData()

        ix = np.where(self.y_pred == 0)
        axis[i, j].scatter(test_Circuit[ix, 0], test_Circuit[ix, 1], c="red", label="0", s=100)
        ix = np.where(self.y_pred == 1)
        axis[i, j].scatter(test_Circuit[ix, 0], test_Circuit[ix, 1], c="green", label="1", s=100)
        axis[i, j].legend()
        axis[i, j].set_title(method)





# A simple Quantum circuit using cirq

input_qubits = cirq.GridQubit.rect(1, 2)  # 1x2 grid.
model_circuit = cirq.Circuit()

alpha1 = sympy.Symbol('a1')
model_circuit.append(cirq.rx(alpha1)(input_qubits[0]))

alpha2 = sympy.Symbol('a2')
model_circuit.append(cirq.rx(alpha2)(input_qubits[1]))

alpha3 = sympy.Symbol('a3')
model_circuit.append(cirq.XX(input_qubits[1],input_qubits[0])**alpha3)

alpha4 = sympy.Symbol('a4')
model_circuit.append(cirq.H(input_qubits[0])**alpha4)

alpha5 = sympy.Symbol('a5')
model_circuit.append(cirq.H(input_qubits[1])**alpha5)

model_readout = [cirq.X(input_qubits[0]),cirq.X(input_qubits[1])]


print(model_circuit)


DataList = ["HLine","VLine", "Triangle", "Circle"]
methodList = ["Ground_Truth","Quantum", "Hybrid", "Single_Dense"]

resultsFolder = "results/"
if not os.path.exists(resultsFolder):
    os.makedirs(resultsFolder)

fig, ax = plt.subplots(2,2,figsize=(16,16))

for data in DataList:
    dataObj = Data_Handler(Nu_Data=15000,pattern=data)
    TFQobj = TFQuantum(dataHandler=dataObj)
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))

    for i, method in enumerate(methodList):
        if method != "Ground_Truth":
            TFQobj.createModel(model_circuit=model_circuit,model_readout=model_readout,type=method)
            TFQobj.train(method=method)
        TFQobj.predict(method=method)
        TFQobj.plot(axis = ax,plotNu=i,method=method)

    plt.savefig(resultsFolder+data+".png")











