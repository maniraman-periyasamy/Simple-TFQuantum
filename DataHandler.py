import numpy as np
from sklearn.model_selection import train_test_split

import cirq
import tensorflow_quantum as tfq
from tensorflow.keras.utils import to_categorical

class Data_Handler:
    """
    Data Handle class create the data with pattern and encodes the same into quantum network.
    """
    def __init__(self, Nu_Data, pattern = "Triangle", test = 0.1, val = 0.1):

        self.Nu_Data = Nu_Data
        self.pattern = pattern
        self.test = 0.1
        self.val = 0.1

        self.get_data()
        self.load_circuit()

    def get_data(self):
        self.data = np.random.uniform(0,1,(self.Nu_Data,2))
        if self.pattern == "VLine":
            self.label = np.where((self.data[:, 0]) <= 0.75, 1, 0)
        elif self.pattern == "HLine":
            self.label = np.where((self.data[:, 1]) <= 0.75, 1, 0)
        elif self.pattern == "Triangle":
            self.label = np.where((self.data[:, 0]+ self.data[:, 1]) <= 1, 1, 0)
        else:
            self.label = np.where(((0.5-self.data[:, 0])**2+ (0.5-self.data[:, 1])**2) <= 0.35**2, 1, 0)

        self.label = to_categorical(self.label)

        self.train_Data, self.test_Data, self.train_label, self.test_label = train_test_split(self.data,self.label,
                                                                                              test_size=0.1,
                                                                                              shuffle = True)
        self.train_Data, self.val_Data, self.train_label, self.val_label = train_test_split(self.train_Data, self.train_label,
                                                                                              test_size=0.1,
                                                                                              shuffle=True)

    def create_circuit(self, train_Data):

        circuit_List = []
        for x in train_Data:
            qubits = cirq.GridQubit.rect(1, 2)
            circuit = cirq.Circuit()
            for i, val in enumerate(x):
                if i == 0:
                    circuit.append(cirq.XPowGate(exponent=val)(qubits[i]))
                if i == 1:
                    circuit.append(cirq.YPowGate(exponent=val)(qubits[i]))
            circuit_List.append(circuit)
        return circuit_List

    def load_circuit(self):

        self.train_circ = self.create_circuit(self.train_Data)
        self.test_circ = self.create_circuit(self.test_Data)
        self.val_circ = self.create_circuit(self.val_Data)

        self.train_circ = tfq.convert_to_tensor(self.train_circ)
        self.test_circ = tfq.convert_to_tensor(self.test_circ)
        self.val_circ = tfq.convert_to_tensor(self.val_circ)

    def get_trainData(self):
        return self.train_Data, self.train_label

    def get_trainCircuit(self):
        return self.train_circ, self.train_label

    def get_testData(self):
        return self.test_Data, self.test_label

    def get_testCircuit(self):
        return self.test_circ, self.test_label

    def get_valData(self):
        return self.val_Data, self.val_label

    def get_valCircuit(self):
        return self.val_circ, self.val_label
