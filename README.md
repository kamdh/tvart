# Time-varying Autoregression with Low Rank Tensors (TVART)
### by Kameron Decker Harris

This is the code repository for the TVART method, to accompany the
paper "Time-varying Autoregression with Low Rank Tensors" by
Kameron Decker Harris, Aleksandr Aravkin, Rajesh Rao, and Bing Brunton.

Dependencies:
* MATLAB R2017b (not tested on earlier versions) with relevant toolboxes
* UNLocBox https://epfl-lts2.github.io/unlocbox-html/
* Python 2.7 or 3.6 with numpy, scipy (optional, for preprocessing neural example)

## src/

The files to run the TVART algorithm and examples are included here.

* TVART_alt_min.m - implementes the alternating minimization algorithm described in the text
* switching_linear.m - switching linear test case
* smooth_linear.m - smooth linear test case
* example_worms.m - worm behavior example
* example_el_nino.m - sea surface temperature example
* preprocess_neurotycho.py - preprocessing script to remove line noise and compute band power for neural example
* example_neurotycho.m - neural activity example
* other files: helper functions, iPython notebooks used to compare with SLDS,
  switching_linear_comparison* and smooth_linear_comparison* run sweeps of test problems across N

## data/

The data for the examples is stored here.
You will need to carry out some extra steps to run all examples:

### Worm behavior

We obtained the code and data from Costa et al. from
https://github.com/AntonioCCosta/local-linear-segmentation.
To just run our example, all that is needed is "worm_tseries.h5".

### Sea surface temperature

In order to run the "Sea surface temperature" example, you must download
* sst.wkmean.1990-present.nc
* lsmask.nc

from  https://www.esrl.noaa.gov/psd/repository/entry/show/PSD+Climate+Data+Repository/Public/PSD+Datasets/NOAA+OI+SST/Weekly+and+Monthly/.

The files "ersst4.nino.mth.81-10.ascii" and "PDO.txt" are from https://www.cpc.ncep.noaa.gov/data/indices/ersst4.nino.mth.81-10.ascii and http://research.jisao.washington.edu/pdo/PDO.latest.txt.

### Neural activity

These data are kindly provided by the [Neurotycho project](http://neurotycho.org):
http://neurotycho.brain.riken.jp/download/base/20090525S1_Food-Tracking_K1_Zenas+Chao_mat_ECoG64-Motion8.zip.

In order to prepare the data for our algorithm, you must run the preprocessing script.

## figures/

After running the code, figures will be saved in this directory.
We include some figures modified from Neurotycho http://neurotycho.org/food-tracking-task.