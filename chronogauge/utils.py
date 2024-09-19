import numpy as np
import pandas as pd
import math

def cyclic_time(times):
    #this is used to convert the target (time of sampling) in hours to cosine and sine values
    times = times % 24
    t_cos = -np.cos((2 * np.pi * times.astype('float64') / 24)+(np.pi/2))
    t_sin = np.sin((2 * np.pi * times.astype('float64') / 24)+(np.pi/2))

    t_circular = np.concatenate((np.asarray(t_cos).reshape(-1, 1), np.asarray(t_sin).reshape(-1, 1)), axis=1)


    return t_circular


def time24(ipreds):
    #returns times as an hourly value within a 24-hour modulus
    preds = []
    for i in range(ipreds.shape[0]):
        preds.append(math.atan2(ipreds[i, 0], ipreds[i, 1]) / math.pi * 12)

    for i in range(len(preds)):
        if preds[i] < 0:
            preds[i] = preds[i] + 24
    return preds

def errors(pred, true):
    #from 24-hour time predictions, get error in minutes
    err = pred - true
    for i in range(0, err.shape[0]):
        if err.iloc[i] > 12:
            err.iloc[i] = err.iloc[i] - 24
        if err.iloc[i] < -12:
            err.iloc[i] = err.iloc[i] + 24
    return err*60

