import numpy as np
import pandas as pd
import os, time

class BQueue:
    
    def __init__(self, seq_len, feat_len):
        """
        :param seq_len: block size
        :param feat_len: size of features
        """
        self._block = np.zeros((seq_len, feat_len))
        self.shape = self._block.shape
    
    def get_block(self):
        return self._block

    def add_row(self, new_row):
        """
        :param new_row: flat ndarray of sizee `feat_len`
        """
        self._block = self.__insert(new_row)

    def __insert(self, new_row):
        """
        insert new row at begining by removing row at end
        :param new_row: flat ndarray of sizee `feat_len`
        """
        shiftup = np.roll(self._block, -1, axis=0)
        shiftup[-1] = new_row
        return shiftup

    def add_multiple_rows(self, rows):
        """
        :param rows: ndarray of shape (n_rows, feat_len)
        """
        for rowidx in range(len(rows)):
            self._block = self.__insert(rows[rowidx])


if __name__ == "__main__":

    print("================================")
    print("+ internal block initially:")
    print("================================")
    seq_len, feat_len = 10, 5
    bq = BQueue(seq_len, feat_len)
    print(bq.get_block())
    print(bq.shape)

    print("================================")
    print("+ insert rows in animation")
    print("================================")
    time.sleep(1)
    os.system('clear')
    ANIM_LEN = 0
    for i in range(ANIM_LEN):
        row_to_add = np.ones(feat_len)*i
        bq.add_row(row_to_add)
        print(bq.get_block())
        time.sleep(1)
        os.system('clear')

    print("====================================")
    print("+ insert multiple rows in animation:")
    print("====================================")
    rows_to_add = np.array([
        np.ones(feat_len) * 1,
        np.ones(feat_len) * 2,
        np.ones(feat_len) * 3,
        np.ones(feat_len) * 4,
    ])
    bq = BQueue(seq_len, feat_len)
    bq.add_multiple_rows(rows_to_add)
    print(bq.get_block())