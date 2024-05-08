Interpolation
=====

.. _interpolation:

Two main types of interpolation:
1. Aligning with standard time grid. This method doesn't increase the number of time steps, but may result in downsampling. The user chooses a time resolution (e.g., nearest 30 minutes). Data are interpolated when within a time step away from the standard time, so 3:35 is interpolated to 3:30 for example, but if the data is two hourly, the aligned data is still two hourly - gaps are not filled.
2. Interpolating to a regular grid. This is the standard kind of interpolation where new data points are estimated in between original data points to follow a new time grid.

There's also the option of more complex interpolation, where a trusted set of reference buoys is used and then the missing data is estimated in anomaly space.