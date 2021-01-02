# class for returning best realtime predictions in a given interval of frames slice
# -------------------------------------------------------------------------------------------------------------
import math

class ClassificationStreamPredictor:
    def __init__(self, interval_size, n_cat):
        """
        :@param n_cat: total classes
        :@param interval_size: the number of consecutive predictions which are condensed into single prediction
        """
        self.interval_size = interval_size
        self.size = 0
        
        self.preds = {} # conf scores as values wrt class idxs
        self.preds_data = {} # data accumulator for future use
        for class_idx in n_cat:
            self.preds[class_idx] = 0
            self.preds_data[class_idx] = []

        self.__empty_preds = self.preds
        self.__empty_preds_data = self.preds_data


    def add_pred(self, pred_class_id, conf, data=None):
        """
        :@param pred_class: integer >=0 and <n_cat 
        :@param conf: confidence score for `pred_class`
        :@param data: predicton data apart from conf and class id
        """
        if self.size > self.interval_size:
            self.__reset_accumulators()
        
        self.preds[pred_class_id] += conf
        self.preds_data[pred_class_id].append(data)
        self.size += 1

    def __reset_accumulators(self):
        self.preds = self.__empty_preds
        self.preds_data = self.__empty_preds_data
        self.size = 0


    @staticmethod
    def __pred_condition(score, best_score, pred_type):
        if pred_type == "highest": return score > best_score
        else: return score < best_score

    def predict(self, pred_type="highest"):
        """ 
        + "highest"
            best is the one with highest conf value. 
            Example, if conf is accuracy, it must be highest
        + "lowest"
            best is the one with least conf value. 
            Example, if conf is similarity, it must be least
        """
        best_score = -math.inf if pred_type == "highest" else math.inf
        best_class = None
        for cls, score in self.preds.items():
            if self.__pred_condition(score, best_score, pred_type):
                best_class = cls
                best_score = score
        return (self.preds[best_class], best_score ,self.preds_data[best_class])


if __name__ == "__main__":
    pass
    # test ClassificationStremPredictor