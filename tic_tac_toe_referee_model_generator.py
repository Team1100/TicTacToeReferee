#!/usr/bin/python

import os
import sys
import csv
import numpy as np
import argparse
import tensorflow as tf
import keras

class TicTacToeReferee(object):
    INPUT_KEYS = ["r1c1","r1c2","r1c3","r2c1","r2c2","r2c3","r3c1","r3c2","r3c3"]
    OUTPUT_KEYS = ["xwon","owon","tie","invalid","incomplete"]
    NUM_EPOCHS=30
    def __init__(self, output_keys=OUTPUT_KEYS, epochs=NUM_EPOCHS):
        self.input_size = 9
        self.hidden_size = 15
        self.output_size = len(output_keys)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_size)),
            tf.keras.layers.Dense(self.hidden_size, activation='relu'),
            tf.keras.layers.Dense(self.hidden_size, activation='relu'),
            tf.keras.layers.Dense(self.output_size, activation='sigmoid'),
            ])
        self.output_keys = output_keys
        self.data_dict = None
        self.input_data = None
        self.output_data = None
        self.epochs = epochs
    def import_training_data(self, filename="TicTacToeBoardClassifierData.csv", output_keys=OUTPUT_KEYS):
        input_keys = self.INPUT_KEYS
        keys = input_keys + self.output_keys
        with open(filename, newline='') as datafile:
            reader = csv.DictReader(datafile)
            self.data_dict = reader
            input_list = []
            output_list = []
            for row in reader:
                input_row = [ float(row[x]) for x in input_keys ]
                input_list.append(input_row)
                output_row = [ float(row[x]) for x in self.output_keys ] 
                output_list.append(output_row)
            self.input_data = tf.constant(input_list, dtype=float)
            print(self.input_data)
            self.output_data = tf.constant(output_list, dtype=float)
            print(self.output_data)
    def train(self):
        self.model.compile(keras.optimizers.SGD(learning_rate=0.8), loss='mean_squared_error', metrics=['accuracy'])
        self.model.fit(self.input_data, self.output_data, batch_size=10, epochs=self.epochs, validation_split=0.05, shuffle=True, verbose=2)
    def save(self, filename):
        self.model.save(filename)

def ticTacToeBoard(s):
    board = [float(v) for v in s.split(',')]
    if len(board) != 9:
        raise Exception("Expected nine entries for tic tac toe board!")
    return board

def main() -> int:
    """Interact with the TicTacToeReferee AI model"""
    ap = argparse.ArgumentParser(description='A programe for interacting with a TicTacToeReferee AI model')
    ap.add_argument("--data-file", default="TicTacToeBoardClassifierData.csv", help="A CSV file containing rows with 9 columns of data input and 5 expected outputs")
    ap.add_argument("-o", "--output", choices=TicTacToeReferee.OUTPUT_KEYS, default=None, action='append', help="The set of outputs to to use when training the model.")
    ap.add_argument("--epochs", type=int, default=TicTacToeReferee.NUM_EPOCHS, help="The number of epochs to use for training.")
    ap.add_argument("--save-file", default="tic_tac_toe_referee.keras", help="The location to save the keras model after training")
    ap.add_argument("--load-file", default="tic_tac_toe_referee.keras", help="The location to load the keras model from to use it")
    ap.add_argument("--action", default='train', choices=['train', 'use'], help="The action to be performed on the model")
    ap.add_argument("--input-board", default=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], type=ticTacToeBoard, help="The board input used in conjunction with the 'use' action")
    pa = ap.parse_args()
    print(pa)
    if pa.action == "train":
        output_keys = TicTacToeReferee.OUTPUT_KEYS
        if pa.output:
            output_keys = pa.output
        referee = TicTacToeReferee(output_keys=output_keys, epochs=pa.epochs)
        referee.import_training_data(filename=pa.data_file)
        referee.train()
        referee.save(pa.save_file)
    else:
        model = keras.models.load_model(pa.load_file)
        in_tensor = tf.constant([pa.input_board], dtype=float)
        print(in_tensor)
        output = model(in_tensor)
        print(output)
    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
