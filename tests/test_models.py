"""Tests for statistics functions within the Model layer."""
from pdb import set_trace

import numpy as np
from numpy import testing as npt
import pytest

from inflammation.models import daily_mean, daily_max, daily_min, patient_normalise

@pytest.mark.parametrize(
    "test, expected, roger_that",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0], "Roger"),
        ([[1, 2], [3, 4], [5, 6]], [3, 4], "Yes Sir!")
    ]
)
def test_daily_mean_with_different_inputs(test, expected, roger_that):
    """ test daily mean with different inputs """
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))
    print(roger_that) # This does nothing for unittest


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]]),
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    ])
def test_normalize_with_different_inputs(test, expected):
    """ test daily mean with different inputs """
    result = patient_normalise(np.array(test))
    print(result)
    # import pdb; pdb,set_trace()
    # breakpoint() # Python function
    npt.assert_allclose(result,
                        np.array(expected),
                        rtol=1E-2,
                        atol=1E-2)


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_max_arange():
    """ Test daily_max """
    # test_input = np.arange(6).reshape(3, 2)
    test_input = np.array([[0, 1],
                           [2, 3],
                           [4, 5]])
    test_result = np.array([4, 5])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)


def test_daily_min_arange():
    """ Test daily_min """
    # test_input = np.arange(6).reshape(3, 2)
    test_input = np.array([[0, 1],
                           [2, 3],
                           [4, 5]])
    test_result = np.array([0, 1])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)


def test_daily_mean_wrong_input():
    """ playing with pytest.raise """
    with pytest.raises(TypeError):
        error_expected = daily_mean([['1', 2], ['A', 'B']])
        # error_expected = daily_mean([['Polly', 'Exparrot'], ['Biggus', 'Dickus']])


def test_normalize_negative_input():
    """ playing with pytest.raise """
    bad_data = [[1, 2, 3], [4, 5, 6], [7, 8, -9]]
    with pytest.raises(ValueError):
        error_expected = patient_normalise(np.array(bad_data))