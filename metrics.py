# Adapted from metrics written by meetshah1995
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

import numpy as np


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        
    def __fast_hist__(self, ground_true, prediction, n_class):
        mask = (ground_true >= 0) & (ground_true < n_class)
        hist = np.bincount(
            n_class * ground_true[mask].astype(int) + prediction[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, ground_trues, predictions):
        for gt, pred in zip(ground_trues, predictions):
            self.confusion_matrix += self.__fast_hist__(gt.flatten(), pred.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        cls_iu = dict(zip(range(self.n_classes), iu))
        return ({'Overall Acc': acc,
                 'Mean Acc': acc_cls,
                 'Mean IoU': mean_iu},
               cls_iu)

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.loss_hist = []

    def reset(self):
        self.loss = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, loss, n=1):
        self.loss = loss
        self.sum += loss * n
        self.count += n
        self.avg = self.sum / self.count
        self.loss_hist.append(self.avg)
