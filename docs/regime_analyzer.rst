Regime Analyzer
===============

.. module:: market_analyzer.regime_analyzer

This module provides tools for detecting and analyzing market regimes using various statistical methods.

Classes
-------

.. autoclass:: RegimeAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~RegimeAnalyzer.fit
      ~RegimeAnalyzer.transform
      ~RegimeAnalyzer.fit_transform

   .. rubric:: Private Methods

   .. autosummary::
      :nosignatures:

      ~RegimeAnalyzer._fit_hmm
      ~RegimeAnalyzer._transform_hmm
      ~RegimeAnalyzer._detect_breaks
      ~RegimeAnalyzer._detect_trend
      ~RegimeAnalyzer._compute_composite_regime

Examples
--------

Here's a basic example of how to use the RegimeAnalyzer class:

.. code-block:: python

    import pandas as pd
    from market_analyzer.regime_analyzer import RegimeAnalyzer

    # Prepare your data
    data = pd.DataFrame({
        'returns': [...],
        'volume': [...],
        'price': [...]
    })

    # Initialize the analyzer
    analyzer = RegimeAnalyzer(
        n_states=3,
        window=252,
        detection_methods=['hmm', 'breaks', 'trend']
    )

    # Fit and transform the data
    regimes = analyzer.fit_transform(data)

    # Print the detected regimes
    print(regimes['composite_regime'])

Notes
-----

The RegimeAnalyzer combines multiple methods for regime detection:

1. Hidden Markov Models (HMM)
   - Uses GaussianHMM to identify distinct market states
   - Provides state probabilities and regime classifications

2. Structural Breaks
   - Detects significant changes in returns, volatility, and volume
   - Uses rolling window analysis with statistical tests

3. Trend Analysis
   - Identifies price trends using moving averages
   - Classifies markets into uptrend/downtrend regimes

See Also
--------

* :class:`sklearn.base.BaseEstimator`
* :class:`sklearn.base.TransformerMixin`
* :class:`hmmlearn.hmm.GaussianHMM`