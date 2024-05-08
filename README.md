# icedrift: a library for cleaning, interpolating, and analyzing drifting sea ice trajectory data
The IceDrift library contains tools for cleaning, regridding, and analysing sea ice trajectory data, in particular data from drifting buoys, ice stations, and tracked ice floes. This library contains tools used to identify questionable data points, interpolate data to fixed grids. The analysis software includes tools to compute common derived properties, such as velocity and acceleration, and more complex measures such as strain rates and dispersion metrics. The code is designed for work with Arctic drifting buoy data such as data from the International Arctic Buoy Program (IABP), the Multidisciplinary Drifting Observatory for the Study of Arctic Climate (MOSAiC), and the Sea Ice Dynamic Experiment (SIDEx). Additional data that can be processed with IceDrift includes tracked floes from the Ice Floe Tracker (IFT).

## cleaning
Outlier detection is a common problem in data analysis and there are numerous sophisticated methods in existence. We focus here on outliers that are detectable by consideration of physics: time moving backwards, drifters repeating the same pattern endlessly, jumps in position that would only be possible with teleportation. The approach at this step is nondestructive: questionable data are flagged, not removed. The process is based on the data processing steps taken by Hutchings and Martini. In each case, a flag of True means that an problem has been identified.

### basic quality control
`check_dates` looks for duplicated dates within a time tolerance, as well as checking for problems in date formats and reversals in time. Default time tolerance is (1 min? 5 min?)

`check_gaps` returns True if there is a set of points that is shorter than a threshold and is isolated by a gap larger than a second threshold.

`check_positions` looks for duplicated lat/lon points as well as points outside the range of valid lat/lon.

### simple outlier detection
`detect_outliers` contains a few methods for outlier detection: z-score, median filters


### complex checks
These tools use statistical methods to identify outliers.


## interpolation
Comparison of multiple trajectories requires alignment on a common time grid. The two main functions in the interpolation module are designed to first interpolate to align data with a regular grid, without filling gaps, then to fill gaps up to a specified length. The default is to interpolate to the nearest 5 minutes. Interpolation uncertainty is estimated by comparing the distance between original position and the interpolated position at the same times, then assigning the absolute value of the nearest direct error measurement to the values on the new grid. 

TBD: show method with a figure

Gap filling methods:
1. 1D interpolation using scipy.interpolate
2. Kriging?


use cubic spline interpolation to align the tracks to a common time grid. By default I only allow gaps to filled that are up to 4 hours long (i.e., for 2 hourly data, only one missing point).

Because the buoys drift as an array, gap filling for winter periods could be done with the regression model approach.

## analysis
Currently limited to calculating velocity and acceleration and reprojecting latitude/longitude onto a Lambert Azimuthal Equal Area projection.

Some ideas to include:
- rotary spectral analysis, to investigate tidal and inertial oscillations  
- dispersion (single, pair, or triplet)  
- deformation (polygons, based on Jenny's approach)
In addition, it contains an implementation of the polygon-based deformation calculation code from Hutchings et al., 2012 and 2018.

# Status
Currently, I'm working on automatically identifying a common problem where one or two datapoints are slightly offset from the main buoy track. I'm trying to initially make it work with only consideration of the buoy track itself rather than using outside information (like wind speed or sea ice concentration). I've gotten it to work pretty well (but not perfectly) for a few buoys at least. 

TBD short term:
- Incorporate the updates from what I did for the MOSAiC data cleaning 
- Set up test cases for each of the outlier check functions