#!/usr/bin/python

import os
import sys
import csv
import numpy as np
import tensorflow as tf

class TicTacToeReferee(object):
    NUM_EPOCHS=5
    def __init__(self):
        self.input_size = 9
        self.hidden_size = 15
        self.output_size = 5
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_size)),
            tf.keras.layers.Dense(self.hidden_size, activation='relu'),
            tf.keras.layers.Dense(self.output_size, activation='relu'),
            ])
        self.data_dict = None
        self.input_data = None
        self.output_data = None
    def import_training_data(self, filename="TicTacToeBoardClassifierData.csv"):
        input_keys = ["r1c1","r1c2","r1c3","r2c1","r2c2","r2c3","r3c1","r3c2","r3c3"]
        output_keys = ["xwon","owon","tie","invalid","incomplete"]
        keys = input_keys + output_keys
        with open(filename, newline='') as datafile:
            reader = csv.DictReader(datafile)
            self.data_dict = reader
            input_list = []
            output_list = []
            for row in reader:
                input_row = [ float(row[x]) for x in input_keys ]
                input_list.append(input_row)
                output_row = [ float(row[x]) for x in output_keys ] 
                output_list.append(output_row)
            self.input_data = tf.constant(input_list, dtype=float)
            print(self.input_data)
            self.output_data = tf.constant(output_list, dtype=float)
            print(self.output_data)
    def train(self, epochs=NUM_EPOCHS):
        self.model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.input_data, self.output_data, batch_size=1, epochs=epochs, validation_split=0.10, shuffle=True, verbose=2)


def main() -> int:
    """Interact with the TicTacToeReferee AI model"""
    referee = TicTacToeReferee()
    referee.import_training_data()
    referee.train()
    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
