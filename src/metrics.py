import numpy as np

def sla_violation(y_true, y_pred, tolerance=0.1):
    violations = ((y_pred + 1e-8) < (1 - tolerance) * y_true).sum()
    return violations / y_true.size

import tensorflow.keras.backend as K

def balanced_sla_loss(y_true, y_pred, tolerance=0.1, under_scale=10.0, over_scale=1.0):
    sla_threshold = (1.0 - tolerance) * y_true

    under = K.relu(sla_threshold - y_pred)       # penalize underprovisioning
    over = K.relu(y_pred - y_true)               # lightly penalize overprovisioning

    loss = under_scale * K.mean(under) + over_scale * K.mean(over)
    return loss
