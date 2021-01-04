# script to read the csv file generated inside `output/real_time_video/<video-name>/posepoints_csv/file.csv`
# and generate class labesl as outputs based on given model...
# -----------------------------------------------------------------------------------------------------------

import json, config, joblib, torch
import pandas as pd
import numpy as np
from model.siamese_lstm import SiameseNetwork, LSTM
from model.defaults import device

from utils.buckets.stream_predictor import ClassificationStreamPredictor
from utils.buckets.block_queue import BQueue

DISTANCE_CLF_PATH = config.DISTANCE_CLASSIFIER
TRAIN_LABELS_PATH = config.TRAIN_LABELS_FOR_IDXS
BEST_MODEL_PATH = config.BEST_MODEL


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

        self.dist_clf = joblib.load(DISTANCE_CLF_PATH)
        self.numpy_labels_train = joblib.load(TRAIN_LABELS_PATH)
        self.model = torch.load(BEST_MODEL_PATH).to(device)
    # ====================================================================
    # end: basic
    # ====================================================================


    # ====================================================================
    # beg: pytorch predictor
    # ====================================================================
    def __siamese_predictor(self, block):
        with torch.no_grad():
            embedding = self.model.forward_once(block)
            dists, train_idxs = self.dist_clf.kneighbors(
                embedding.data.cpu().numpy())
            predicted_ids = self.numpy_labels_train[train_idxs.flatten()]
            return (dists, predicted_ids)
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
        dists, preds = self.__siamese_predictor(block)
        return (dists, preds)
    # ====================================================================
    # end: row-wise inference 
    # ====================================================================


if __name__ == "__main__":
    REAL_TIME_VIDEO_NAME = "TestVideo.mp4" # Name of folder to process inisde `output/real_time_video/`

    path = CONFIG.REAL_TIME_VIDEOS_OUTPUT_DIR + REAL_TIME_VIDEO_NAME
    processor = ProcessPosePointsJSON(path)
    
    for frame_num in [30]:
        ret = processor.predict(10)
        print(f"{frame_num} \tret")