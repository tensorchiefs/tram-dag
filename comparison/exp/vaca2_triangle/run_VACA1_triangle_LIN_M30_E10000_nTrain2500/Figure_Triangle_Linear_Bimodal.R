######################################################
# This is the Vaca1 Linear Triangle Experiment 
#


####################################
# Setting the "global" configurations 
library(tidyverse)
library(ggpubr)
source('R/tram_dag/utils.R')
library(R.utils)
DEBUG = FALSE
DEBUG_NO_EXTRA = FALSE
USE_EXTERNAL_DATA = FALSE 

#### Parameters influencing he training ######
M = 30

VACA = 1L 
#1 Use the DGP as described in the original VACA paper
#2 Use the DGP as described in the preprint VACA2 (no bimodal X1)

TYPE = "LIN" # LIN using linear version "NONLIN" using the non-linear version
EPOCHS = 10000
nTrain = 2500L
SUFFIX = sprintf("run_VACA%d_triangle_%s_M%d_E%d_nTrain%d", VACA, TYPE, M, EPOCHS, nTrain)

DROPBOX = 'C:/Users/sick/dl Dropbox/beate sick/IDP_Projekte/DL_Projekte/shared_Oliver_Beate/Causality_2022/tram_DAG/'
DROPBOX = '~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/Causality_2022/tram_DAG/'
if (rstudioapi::isAvailable()) {
  context <- rstudioapi::getSourceEditorContext()
  this_file <- context$path
  print(this_file)
} else{
  this_file = "~/Documents/GitHub/causality/R/tram_dag/vaca_triangle.r"
}

#### Parameters influencing the prediction ####################
DoX = 1 #The variable on which the do-intervention should occur
DoX = 2

source('R/tram_dag/vaca_triangle.r')
