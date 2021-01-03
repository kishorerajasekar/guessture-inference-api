# script to read the csv file generated inside `output/real_time_video/<video-name>/posepoints_csv/file.csv`
# and generate class labesl as outputs based on given model...
# -----------------------------------------------------------------------------------------------------------

import json, config
import pandas as pd
import numpy as np
from utils.buckets.stream_predictor import ClassificationStreamPredictor
from utils.buckets.block_queue import BQueue


REAL_TIME_VIDEO_NAME = None # Name of folder to process inisde `output/real_time_video/`

class ProcessPosePointsJSON:
    # ====================================================================
    # beg: basic 
    # ====================================================================
    def __init__(filepath, seq_len=90, feat_len=110):
        self.filepath = filepath
        self.video_pose_points = pd.DataFrame(
            json.load(
                open(filepath + "/posepoints_json/data.json"))).values
        self.cur_row_idx = -1
        self.seq_len = seq_len
        self.feat_len = feat_len
        self.bqueue = BQueue(self.seq_len, self.feat_len)
    # ====================================================================
    # end: basic
    # ====================================================================


    # ====================================================================
    # beg: pytorch predictor
    # ====================================================================
    def __predict_classes_and_conf(self, block, interva_size=40):
        """
        """
        pass
    # ====================================================================
    # end: pytorch predictor
    # ====================================================================
 

    # ====================================================================
    # beg: continous inference 
    # ====================================================================
    # use these methods to predict continously without 
    # any request from frontend after each yielded prediction
    def get_next_frame_poseponits(self):
        """ returns ndarray of size (feal_ten,) """
        self.cur_row_idx += 1
        yield self.video_pose_points[self.cur_row_idx]

    def start_inference(self):
        for posepoints in self.get_next_frame_poseponits():
            self.bqueue.add_row(posepoints)
            # get precicted class and confidence
            # yield results
    # ====================================================================
    # end: continous inference 
    # ====================================================================


    # ====================================================================
    # beg: row-wise inference 
    # ====================================================================
    # use these methods to predict only upon request from 
    # frontend.
    def get_seq_upto(self, last_framenum_in_seq):
        
        end_seq_idx = last_framenum_in_seq
        beg_seq_idx = last_framenum_in_seq - self.seq_len
        beg_seq_idx = beg_seq_idx if beg_seq_idx >= 0 else 0
        
        self.bqueue.add_multiple_rows(
            self.video_pose_points[beg_seq_idx:end_seq_idx])
        return self.bqueue.get_block()

    def predict(frame_num):
        """ frame num is same as index in csv"""
        block = self.get_seq_upto(frame_num)
        # pred from block
        # return preds
    # ====================================================================
    # end: row-wise inference 
    # ====================================================================


if __name__ == "__main__":
    path = CONFIG.REAL_TIME_VIDEOS_OUTPUT_DIR + REAL_TIME_VIDEO_NAME
    processor = ProcessPosePointsJSON(path)