# script to read the csv file generated inside `output/real_time_video/<video-name>/posepoints_csv/file.csv`
# and generate class labesl as outputs based on given model...
# -----------------------------------------------------------------------------------------------------------

import pandas as pd
import json, config


REAL_TIME_VIDEO_NAME = None # Name of folder to process inisde `output/real_time_video/`

class ProcessPosePointsCSV:
    # ==========================================================
    # beg: basic 
    # ==========================================================
    def __init__(filepath, seq_len=90):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.cur_row_idx = -1
        self.seq_len = seq_len
    # ==========================================================
    # end: basic
    # ==========================================================


    # ==========================================================
    # beg: continous inference 
    # ==========================================================
    # use these methods to predict continously without 
    # any request from frontend after each yielded prediction
    def get_next_frame_poseponits(self):
        self.cur_row_idx += 1
        yield json.loads(self.df.json[self.cur_row_idx])

    def start_inference(self):
        for posepoints in self.get_next_frame_poseponits():
            # load in queue
            # get precicted class and confidence
            # yield results
    # ==========================================================
    # end: continous inference 
    # ==========================================================


    # ==========================================================
    # beg: row-wise inference 
    # ==========================================================
    def get_seq_upto(self, last_framenum_in_seq):
        
        end_seq_idx = last_framenum_in_seq
        beg_seq_idx = last_framenum_in_seq - self.seq_len
        
        if beg_seq_idx >= 0:
            # create queue with block
        else:
            # create pdded queue with block
        # return queue

    def predict(frame_num):
        """ frame num is same as index in csv"""
        block = self.get_seq_upto(frame_num)
        # pred from block
        # return preds
    # ==========================================================
    # end: row-wise inference 
    # ==========================================================
            


if __name__ == "__main__":
    path = CONFIG.REAL_TIME_VIDEOS_OUTPUT_DIR + REAL_TIME_VIDEO_NAME
    processor = ProcessPosePointsCSV(path)