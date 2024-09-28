import autograd.numpy as np

EPS = 1e-15


def unhot(function):
    """Convert one-hot representation into one column."""

    def wrapper(actual, predicted):
        if len(actual.shape) > 1 and actual.shape[1] > 1:
            actual = actual.argmax(axis=1)
        if len(predicted.shape) > 1 and predicted.shape[1] > 1:
            predicted = predicted.argmax(axis=1)
        return function(actual, predicted)

    return wrapper


def absolute_error(actual, predicted):
    return np.abs(actual - predicted)


@unhot
def classification_error(actual, predicted):
    return (actual != predicted).sum() / float(actual.shape[0])


@unhot
def accuracy(actual, predicted):
    return 1.0 - classification_error(actual, predicted)


def mean_absolute_error(actual, predicted):
    return np.mean(absolute_error(actual, predicted))


def squared_error(actual, predicted):
    return (actual - predicted) ** 2


def squared_log_error(actual, predicted):
    return (np.log(np.array(actual) + 1) - np.log(np.array(predicted) + 1)) ** 2


def mean_squared_log_error(actual, predicted):
    return np.mean(squared_log_error(actual, predicted))


def mean_squared_error(actual, predicted):
    return np.mean(squared_error(actual, predicted))


def root_mean_squared_error(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


def root_mean_squared_log_error(actual, predicted):
    return np.sqrt(mean_squared_log_error(actual, predicted))


def logloss(actual, predicted):
    predicted = np.clip(predicted, EPS, 1 - EPS)
    loss = -np.sum(actual * np.log(predicted))
    return loss / float(actual.shape[0])


def hinge(actual, predicted):
    return np.mean(np.max(1.0 - actual * predicted, 0.0))


def binary_crossentropy(actual, predicted):
    predicted = np.clip(predicted, EPS, 1 - EPS)
    return np.mean(-np.sum(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)))

def mean_absolute_percentage_error(actual, predicted):
    return np.mean(np.abs((actual - predicted) / (actual + EPS))) * 100

def r_squared(actual, predicted):
    ss_total = np.sum((actual - np.mean(actual)) ** 2)
    ss_residual = np.sum((actual - predicted) ** 2)
    return 1 - (ss_residual / ss_total)

@unhot
def recall(actual, predicted):
    tp = np.sum((actual == 1) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

@unhot
def precision(actual, predicted):
    tp = np.sum((actual == 1) & (predicted == 1))
    fp = np.sum((actual == 0) & (predicted == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

@unhot
def f1_score(actual, predicted):
    tp = np.sum((actual == 1) & (predicted == 1))
    fp = np.sum((actual == 0) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# aliases
mse = mean_squared_error
rmse = root_mean_squared_error
mae = mean_absolute_error
mape= mean_absolute_percentage_error


def get_metric(name):
    """Return metric function by name"""
    try:
        return globals()[name]
    except Exception:
        raise ValueError("Invalid metric function.")
