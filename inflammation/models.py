"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=",")


def daily_mean(data):
    """
    Calculate the daily mean of a 2d inflammation data array.
    :param data: numpy array or iterable with numbers
    :returns: numpy float or array (depending on shape of data)
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """
    Calculate the daily max of a 2d inflammation data array.
    :param data: numpy array or iterable with numbers
    :returns: numpy float or array (depending on shape of data)
    """
    return np.max(data, axis=0)


def daily_min(data):
    """
    Calculate the daily min of a 2d inflammation data array.
    :param data: numpy array or iterable with numbers
    :returns: numpy float or array (depending on shape of data)
    """
    return np.min(data, axis=0)


def daily_median(data):
    """
    Calculate the daily median of a 2d inflammation data array.
    :param data: numpy array or iterable with numbers
    :returns: numpy float or array (depending on shape of data)
    """
    return np.median(data, axis=0)


def all_negatives_must_die(data):
    """ Negative values? Bye, sayonara, get lost, blah blah blah"""
    return np.any(data < 0)


def patient_normalise(data):
    """Normalize patient data from a 2D array"""
    ans = np.max(data, axis=1)
    if all_negatives_must_die(data):
        raise ValueError("Negatives detected; thou shall not pass.")
    with np.errstate(invalid="ignore", divide="ignore"):
        ans2 = data / ans[:, np.newaxis]
    ans2[np.isnan(ans2)] = 0.0
    return ans2
