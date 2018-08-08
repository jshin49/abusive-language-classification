from sklearn.metrics import precision_recall_fscore_support, classification_report
import numpy as np


class ClassificationReport(object):

    def __init__(self, model, classes, eval_y, preds):
        self.model = model
        self.classes = classes
        self.truth = eval_y
        self.preds = preds

    def on_epoch_end(self, epoch, logs={}):
        """
        F1 = 2 * (precision * recall) / (precision + recall)
        """
        print("Generating Classification Report for epoch: ", epoch)
        print("\n%s\n" % classification_report(
            self.truth, self.preds, target_names=self.classes, digits=3))
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.truth, self.preds)
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            self.truth, self.preds, average='weighted')

        return precision, recall, f1, precision_avg, recall_avg, f1_avg
