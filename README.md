# drifter: a library for cleaning and interpolating drifting buoy data
GPS buoys are a crucial tool for observing sea ice motion. As with any observational tool, it's important to clean the reported buoy tracks before using them in analysis. This library contains tools used to identify questionable data points, interpolate data to fixed grids, as well as tools to compute common properties, such as velocity and acceleration. The code is designed for work with Arctic drifting buoy data, in particular data from the International Arctic Buoy Program, but may be useful for other contexts as well.

## data cleaning
Outlier detection is a common problem in data analysis and there are numerous sophisticated methods in existence. We focus here on outliers that are detectable by consideration of physics: time moving backwards, drifters repeating the same pattern endlessly, jumps in position that would only be possible with teleportation. The approach here is nondestructive: data are not regridded, and "bad" data are flagged, not removed

