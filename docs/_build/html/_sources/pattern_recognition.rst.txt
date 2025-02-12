Pattern Recognition
===================

.. automodule:: market_analyzer.pattern_recognition
   :members:
   :undoc-members:
   :show-inheritance:

Technical Pattern Analysis
--------------------------

This module provides tools for identifying and analyzing technical patterns in financial market data. The key components include:

Data Classes
~~~~~~~~~~~~

- ``TechnicalPattern``: Core class for representing detected patterns
- ``HeadAndShouldersPoints``: Specific points forming a head and shoulders pattern
- ``PatternValidation``: Results of pattern validation checks
- ``VolumePatternType``: Enumeration of volume-price relationship patterns
- ``VolumePattern``: Volume-specific pattern information

Pattern Recognition
~~~~~~~~~~~~~~~~~~~

The ``PatternRecognition`` class implements detection algorithms for common technical patterns:

- Head and Shoulders
- Double Bottom
- Volume-Price Relationships

Lead-Lag Analysis
~~~~~~~~~~~~~~~~~

The ``LeadLagAnalyzer`` class provides tools for analyzing relationships between securities:

- Cross-correlation analysis
- Granger causality testing
- Market leadership detection
- Relationship network analysis