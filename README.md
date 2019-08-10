# Decorr_Index
Moving window de-correlation for Characteristic Repeating Earthquakes (CREs)

[![DOI](https://zenodo.org/badge/81490154.svg)](https://zenodo.org/badge/latestdoi/81490154)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
![alt text](https://www.gnu.org/graphics/lgplv3-147x51.png)

![alt text](https://github.com/echavess/Decorr_Index/blob/master/Decorrelation_Index_Bp_2_20.png)


* How to run the code? :

    ```
    # ----------------I N P U T   P A R A M S----------------------- ###

    bandpass_filter=[2, 20]
    trace_len=[0.6, 10.4]
    windowing = [1, 0.10]

    # ----------------D A T A   L I S T S----------------------- ###
    reference = 'Highly_rep/OCM.HHZ.2019.108.20.27.15' # This is the reference CRE
    data_stream = glob.glob('Highly_rep/*HH*') # List with all the Traces (Vertical components)

    # ----------------R U N  T H E  C O D E----------------------- ###
    decorr_index(reference=reference, stream_waves=data_stream, 
           trace_len=trace_len, bandpass_filter=bandpass_filter, 
           windowing=windowing)
    ```
