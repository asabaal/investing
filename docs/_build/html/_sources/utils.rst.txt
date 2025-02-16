Utils
=====

.. currentmodule:: market_analyzer.utils

This module provides functions for calculating rolling window statistics with a growing window size for initial periods.

Functions
---------

.. autofunction:: market_analyzer.utils.rolling_growing_window

Specialized Growing Window Functions
------------------------------------

The following functions are specialized versions of rolling_growing_window for common statistical operations:

.. autofunction:: market_analyzer.utils.growing_mean

.. autofunction:: market_analyzer.utils.growing_std

.. autofunction:: market_analyzer.utils.growing_sum

.. autofunction:: market_analyzer.utils.growing_max

.. autofunction:: market_analyzer.utils.growing_min

Examples
--------

Basic usage with growing mean:

.. code-block:: python

    import pandas as pd
    from market_analyzer.utils import growing_mean

    # Create sample data
    data = pd.Series([1, 2, 3, 4, 5])
    
    # Calculate growing window mean with window size 3
    result = growing_mean(data, window=3)
    # result: array([1., 1.5, 2., 3., 4.])

Custom function with growing window:

.. code-block:: python

    import numpy as np
    from market_analyzer.utils import rolling_growing_window

    # Define custom weighted mean function
    def weighted_mean(x):
        weights = np.arange(1, len(x) + 1)
        return np.average(x, weights=weights)
    
    # Apply with growing window
    result = rolling_growing_window(data, window=3, func=weighted_mean)

Notes
-----

The growing window functions handle initial periods differently than standard rolling windows:

- For i < window: Uses all available data points up to position i
- For i >= window: Uses standard rolling window of size 'window'

This approach eliminates the need for special handling of NaN values in the initial periods
while maintaining the benefits of a rolling window for later periods.