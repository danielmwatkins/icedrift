Cleaning
=====

.. _cleaning:

Outlier detection is a common problem in data analysis and there are numerous sophisticated methods in existence. We focus here on outliers that are detectable by consideration of physics: time moving backwards, drifters repeating the same pattern endlessly, jumps in position that would only be possible with teleportation. The approach at this step is nondestructive: questionable data are flagged, not removed. The process is based on the data processing steps taken by Hutchings and Martini. In each case, a flag of True means that an problem has been identified.

Basic quality control
----------------
.. autofunction:: icedrift.check_dates looks for duplicated dates within a time tolerance, as well as checking for problems in date formats and reversals in time. Default time tolerance is (1 min? 5 min?)

.. autofunction:: icedrift.check_gaps returns True if there is a set of points that is shorter than a threshold and is isolated by a gap larger than a second threshold.

.. autofunction:: icedrift.check_positions looks for duplicated lat/lon points as well as points outside the range of valid lat/lon.


>>> import icedrift
>>> data = get_test_data_TBD()
>>> flagged_data = icedrift.check_dates(data)


Simple outlier detection
----------------
- Step by step threshold test
- Z-score

Complex outlier detection
----------------
These tools use statistical methods to identify outliers.
- LOESS method
- Multi-trajectory approaches


