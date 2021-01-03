# class for returning best realtime predictions in a given interval of frames slice.
# Instead of predicting 1000 times, predicts samll number (1000/interval_size) of times.
# -------------------------------------------------------------------------------------------------------------
import math

class ClassificationStreamPredictor:
    def __init__(self, interval_size, n_cat, pred_type="lowest"):
        """
        :@param n_cat: total classes
        :@param interval_size: the number of consecutive predictions which are condensed into single prediction
        :@param pred_type:
            + "highest"
                best is the one with highest conf value. 
                Example, if conf is accuracy, it must be highest
            + "lowest"
                best is the one with least conf value. 
                Example, if conf is distance, it must be least
        """
        self.interval_size = interval_size
        self.size = 0
        
        self.preds = {} # conf scores as values wrt class idxs
        self.class_cnts = {} # for normalizing conf for relative measure
        self.preds_data = {} # data accumulator for future use
        for class_idx in range(n_cat):
            self.preds[class_idx] = 0
            self.class_cnts[class_idx] = 0
            self.preds_data[class_idx] = []

        self.__empty_preds = self.preds
        self.__empty_preds_data = self.preds_data
        
        # to predict by aggregated relative conf
        self.pred_type = pred_type
        self.best_score = -math.inf if pred_type == "highest" else math.inf
        self.best_class = None
        # to predict by num of appearances only
        self.class_appearances = []

    def add_pred(self, pred_class_id, conf, data=None):
        """
        :@param pred_class: integer >=0 and <n_cat 
        :@param conf: confidence score for `pred_class`
        :@param data: predicton data apart from conf and class id
        """
        if self.size > self.interval_size:
            self.__reset_accumulators()
        
        self.preds[pred_class_id] += conf
        self.class_cnts[pred_class_id] += 1
        self.preds_data[pred_class_id].append(data)
        self.size += 1
        # for predicting by appearances
        self.class_appearances.append(pred_class_id)

        relative_score = self.preds[pred_class_id]/self.class_cnts[pred_class_id]
        if self.__pred_condition(relative_score, self.best_score, self.pred_type):
            # to predict by aggregated relative conf
            self.best_score = relative_score
            self.best_class = pred_class_id
             

    def __reset_accumulators(self):
        self.preds = self.__empty_preds
        self.preds_data = self.__empty_preds_data
        self.size = 0
        # for predicting by relative scores
        self.best_score = -math.inf if self.pred_type == "highest" else math.inf
        self.best_class = None
        # for predicting by appearances
        self.class_appearances = []        


    @staticmethod
    def __pred_condition(score, best_score, pred_type):
        if pred_type == "highest": return score > best_score
        else: return score < best_score

    def predict(self):
        """ predict using relative scores """
        return (self.best_class, self.best_score, self.preds_data[self.best_class])

    def get_most_frequent_in_interval(self):
        """ predict the most appeared """
        most_frq_cls_id = max(set(self.class_appearances), key = self.class_appearances.count)
        return (most_frq_cls_id, None, self.preds_data[self.best_class])


if __name__ == "__main__":
    
    interval_size, n_cat = 4, 3
    predictor = ClassificationStreamPredictor(interval_size, n_cat, "lowest")

    stream_data_pred_classes          = [0,0,0,0,          1,1,2,1,          2,1,1,1,         2,1,0,0,             0,1,]
    stream_data_pred_dists            = [0.1,0.1,0.7,0.4,  0.3,0.5,0.2,0.6,  0.1,0.5,0.8,0,   0.3,0.3,0.3,0.2,     0,1,]
    # Must predict (relative scores)  :  0                 1                 2                0                    <Nothing>
    # Must predict (most frequent)    :  0                 1                 1                0                    <Nothing>                 

    idx = -1
    for pred_class_id, dist_conf in zip(stream_data_pred_classes, stream_data_pred_dists):
        idx += 1
        predictor.add_pred(pred_class_id, dist_conf)
        
        # Nore: ignore idx zero
        if idx%interval_size == 0:
            print("-------------------------------------------------------------------------------------")
            cur_data = list(zip(stream_data_pred_classes, stream_data_pred_dists))[idx-interval_size:idx]
            print(f"CURRENT DATA: {cur_data} {'IGNORE!!!' if cur_data == [] else ''}")
            print("-------------------------------------------------------------------------------------")
            print(f"\tMost Freq Predicted @streamidx{idx}:", predictor.get_most_frequent_in_interval()[0])
            print(f"\tBest Relative Score @streamidx{idx}:", predictor.predict()[0])
