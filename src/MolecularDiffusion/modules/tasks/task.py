from collections.abc import Mapping, Sequence

from torch import nn



_criterion_name = {
    "mse": "mean squared error",
    "mae": "mean absolute error",
    "bce": "binary cross entropy",
    "ce": "cross entropy",
}

_metric_name = {
    "mae": "mean absolute error",
    "mse": "mean squared error",
    "rmse": "root mean squared error",
    "acc": "accuracy",
    "mcc": "matthews correlation coefficient",
}



def _get_criterion_name(criterion):
    if criterion in _criterion_name:
        return _criterion_name[criterion]
    return "%s loss" % criterion


def _get_metric_name(metric):
    if metric in _metric_name:
        return _metric_name[metric]
    return metric


class Task(nn.Module):

    _option_members = set()

    def _standarize_option(self, x, name):
        if x is None:
            x = {}
        elif isinstance(x, str):
            x = {x: 1}
        elif isinstance(x, Sequence):
            x = dict.fromkeys(x, 1)
        elif not isinstance(x, Mapping):
            raise ValueError("Invalid value `%s` for option member `%s`" % (x, name))
        return x

    def __setattr__(self, key, value):
        if key in self._option_members:
            value = self._standarize_option(value, key)
        super(Task, self).__setattr__(key, value)

    def preprocess(self, train_set, valid_set=None, test_set=None):
        pass

    def predict_and_target(self, batch, all_loss=None, metric=None):
        return self.predict(batch, all_loss, metric), self.target(batch)

    def predict(self, batch, all_loss=None, metric=None):
        raise NotImplementedError

    def target(self, batch):
        raise NotImplementedError

    def evaluate(self, pred, target):
        raise NotImplementedError

