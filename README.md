# Time-varying Autoregression with Low Rank Tensors (TVART)
### by Kameron Decker Harris

This is the code repository for the TVART method, to accompany the
paper "Time-varying Autoregression with Low Rank Tensors" by
Kameron Decker Harris, Aleksandr Aravkin, Rajesh Rao, and Bing Brunton.

## src/

The files to run the TVART algorithm and examples are included here.

* TVART_alt_min.m - main algorithm

## data/

The data for the examples is stored here.
You will need to carry out some extra steps to run all examples:

### Worm behavior

We obtained the code and data from Costa et al. from
https://github.com/AntonioCCosta/local-linear-segmentation.
Specifically, all that is needed is "worm_tseries.h5".

### Sea surface temperature

In order to run the "Sea surface temperature" example, you must download
* sst.wkmean.1990-present.nc from https://www.esrl.noaa.gov/psd/repository/entry/show/PSD+Climate+Data+Repository/Public/PSD+Datasets/NOAA+OI+SST/Weekly+and+Monthly/sst.wkmean.1990-present.nc?entryid=12159560-ab82-48a1-b3e4-88ace20475cd&output=default.html
