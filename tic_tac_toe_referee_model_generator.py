#!/usr/bin/python

import os
import sys
import numpy as np
import tensorflow as tf

class TicTacToeReferee(object):
    NUM_EPOCHS=5
    def __init__(self):
        input_size = 9
        hidden_size = 15
        output_size = 5
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_size)),
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dense(output_size, activation='relu'),
            ])
        self.train_data = []
        self.test_data = []
    def import_training_data(self):
        pass
    def train(self, epochs=self.NUM_EPOCHS):
        self.model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.train_data, epochs, validation_data=([], []), verbose=2)


def main() -> int:
    """Interact with the TicTacToeReferee AI model"""
    referee = TicTacToeReferee()
    referee.train()
    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
