#!/bin/bash

import csv
import sys

class TicTacToeDataGenerator(object):
    def __init__(self):
        self.input_keys = ["r1c1","r1c2","r1c3","r2c1","r2c2","r2c3","r3c1","r3c2","r3c3"]
        self.output_keys = ["xwon","owon","tie","invalid","incomplete"]
        self.fieldnames = self.input_keys + self.output_keys
    def generate_data(self, outfile="TicTacToeBoardClassifierAllData.csv"):
        with open(outfile, 'w', newline='') as datafile:
            writer = csv.DictWriter(datafile, fieldnames=self.fieldnames)

            writer.writeheader()
            # Add 9 for loops iterating through 3 values
            for i1 in [0,0.5,1]:
                for i2 in [0,0.5,1]:
                    for i3 in [0,0.5,1]:
                        for i4 in [0,0.5,1]:
                            for i5 in [0,0.5,1]:
                                for i6 in [0,0.5,1]:
                                    for i7 in [0,0.5,1]:
                                        for i8 in [0,0.5,1]:
                                            for i9 in [0,0.5,1]:
                                                self.write_row(writer, (i1, i2, i3, i4, i5, i6, i7, i8, i9))
    def write_row(self, writer, values):
        assert len(values) == 9
        out_row = {x:y for x,y in zip(self.input_keys, values)}
        b = values
        # 0 1 2
        # 3 4 5
        # 6 7 8
        x = 1
        o = 0
        s = 0.5
        xwon = (b[0] == x and b[1] == x and b[2] == x) or \
               (b[3] == x and b[4] == x and b[5] == x) or \
               (b[6] == x and b[7] == x and b[8] == x) or \
               (b[0] == x and b[3] == x and b[6] == x) or \
               (b[1] == x and b[4] == x and b[7] == x) or \
               (b[2] == x and b[5] == x and b[8] == x) or \
               (b[0] == x and b[4] == x and b[8] == x) or \
               (b[2] == x and b[4] == x and b[6] == x)
        owon = (b[0] == o and b[1] == o and b[2] == o) or \
               (b[3] == o and b[4] == o and b[5] == o) or \
               (b[6] == o and b[7] == o and b[8] == o) or \
               (b[0] == o and b[3] == o and b[6] == o) or \
               (b[1] == o and b[4] == o and b[7] == o) or \
               (b[2] == o and b[5] == o and b[8] == o) or \
               (b[0] == o and b[4] == o and b[8] == o) or \
               (b[2] == o and b[4] == o and b[6] == o)
        tie = all([v in [x,o] for v in b]) and not xwon and not owon
        invalid = (abs(b.count(x) - b.count(o)) > 1) or (xwon and owon)
        incomplete = any([v == s for v in b]) and not xwon and not owon
        out_row["xwon"] = int(xwon)
        out_row["owon"] = int(owon)
        out_row["tie"] = int(tie)
        out_row["invalid"] = int(invalid)
        out_row["incomplete"] = int(incomplete)
        writer.writerow(out_row)

def main() -> int:
    """Generate data for the TicTacToeReferee AI model"""
    generator = TicTacToeDataGenerator()
    generator.generate_data()
    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
