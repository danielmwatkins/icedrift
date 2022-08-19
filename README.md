# drifter: a library for cleaning, interpolating, and analyzing drifting buoy data
GPS buoys are a crucial tool for observing sea ice motion. As with any observational tool, it's important to clean the reported buoy tracks before using them in analysis. This library contains tools used to identify questionable data points, interpolate data to fixed grids, as well as tools to compute common properties, such as velocity and acceleration. The code is designed for work with Arctic drifting buoy data, in particular data from the International Arctic Buoy Program, but may be useful for other contexts as well.

## data cleaning
Outlier detection is a common problem in data analysis and there are numerous sophisticated methods in existence. We focus here on outliers that are detectable by consideration of physics: time moving backwards, drifters repeating the same pattern endlessly, jumps in position that would only be possible with teleportation. The approach at this step is nondestructive: questionable data are flagged, not removed. The process is based on the data processing steps taken by Hutchings and Martini but is in general more conservative.

`flag_duplicates` returns True if the date, latitude, or longitude are duplicated. It checks for duplicated pairs including nonadjacent pairs, and for repeated latitude or longitude values.

`check_times` returns True if time runs backward. TBD: identify if time is outside the range indicated by the file.

`check_speed` returns True if the buoy velocity is higher than a threshold of 1.5 m/s.

### extra cleaning things
Some buoys report data with very high noise, making any kind of analysis dicey. Buoy 3690 for year 2010 is used as an example for dates that are out of order--it has a few points that are randomly in 2007 with longitudes way out of plausible range--but it's got more problems than that. The data is noisy enough that a coherent track only appears in averages over many hours, perhaps up to a day. There should be a tool that estimates the position noise in the buoy track.

## interpolation
Comparison of multiple buoy tracks requires alignment on a common time grid. I use cubic splines and only allow gaps to filled that are up to 4 hours long (i.e., for 2 hourly data, only one missing point). Gap filling for some periods could be done with the regression model approach.

## analysis
Some ideas to include:
- rotary spectral analysis, to investigate tidal and inertial oscillations  
- dispersion (single, pair, or triplet)  
- deformation (polygons, based on Jenny's approach)
