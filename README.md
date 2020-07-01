# Fannie Mae mortgage data

## About 

This project takes loan-level performance data from [FannieMae](https://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html) (FNMA) to predict prepayment. 
We first start with the combined data from FNMA and tide it into a more usable format. We then make the necessary dummy variables and code in the required format to apply 
neural networks to try and predict the month of prepayment.

## Requirements

We give the user the consolidated data from the [FNMA repositories](https://loanperformancedata.fanniemae.com/lppub/index.html). The code is run on Python with the basic packages
so no extra stuff is needed.

Python libraries required
* pip install pandas
* pip install numpy
* pip install tensorflow

## Usage

The code is fairly straightforward. The main objective of the work is to provide an application of Tensorflow's neural network with it's correct calibration and data preparation.
And it's later visualization using Tensorboard. 

## Roadmap

We will later add the merging and work behind to obtain the Combined_Data files. The code for this comes from FNMA, but since it is only written in R, we will transform it into
an usable python code. This will be useful for those who want to work with this data, for different periods of time or adapt the code to get some variables you may want that 
weren't included.




