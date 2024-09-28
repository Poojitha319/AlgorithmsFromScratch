from metrics import mse, logloss, mae, hinge, binary_crossentropy
categorical_crossentropy = logloss
def get_loss(name):
    try:
        return globals()[name]
    except KeyError:
        raise ValueError("Invalid metric function.")
