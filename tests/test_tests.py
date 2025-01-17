import numpy as np
import numpy.testing as npt

import inflammation.models

test_inputs = np.zeros((2,2))
test_output = np.zeros(2)

npt.assert_array_equal(inflammation.models.daily_mean(test_inputs), test_output)
