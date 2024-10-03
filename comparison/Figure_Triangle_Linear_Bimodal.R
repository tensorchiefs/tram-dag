#### Local Configuration to savely load tensorflow and tfp ####
##### Oliver's MAC ####
reticulate::use_python("/Users/oli/miniforge3/envs/r-tensorflow/bin/python3.8", required = TRUE)
library(reticulate)
reticulate::py_config()
library(tfprobability)


######################################################
# This is the Vaca1 Linear Triangle Experiment 

####################################
# Setting the "global" configurations 
library(tidyverse)
library(ggpubr)
library(gridExtra)
library(grid)

source('comparison/utils.R')

version_info()
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

DROPBOX = '~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/shared_Oliver_Beate/Causality_2022/tram_DAG/' #Location of data and trained weights
DROPBOX = 'comparison/' #Location of data and trained weights

if (rstudioapi::isAvailable()) {
  context <- rstudioapi::getSourceEditorContext()
  this_file <- context$path
  print(this_file)
} else{
  this_file = "~/Documents/GitHub/tram_DAG/Figure_Triangle_Linear_Bimodal.R"
}

#### Parameters influencing the prediction ####################
DoX = 1 #The variable on which the do-intervention should occur
DoX = 2

#Possible 
source('comparison/vaca_triangle.r')

######### Creating the plots #######

######## L1 Observed Plotting the Observed Data and Fits #####
Xmodel = unscale(train$df_orig, sampleObs(thetaNN_l, A=train$A, 25000))$numpy()
XDGP = dgp(25000, dat_train = train$df_orig)$df_orig$numpy()
#Xref <- as.matrix(read_csv("data/VACA1_triangle_lin/carefl/VACA1_triangle_LIN_XobsModel.csv", col_names = FALSE))
  
## Execute for NSF
Xref <- as.matrix(read_csv("data/VACA1_triangle_lin/NSF/VACA1_triangle_LIN_XobsModel.csv", col_names = FALSE))
names <- c("Ours", "NSF", "DGP")  
custom_colors <- c("Ours" = "#1E88E5", "NSF" = "#FFC107", "DGP" = "red") 

Xref <- as.matrix(read_csv("data/VACA1_triangle_lin/25K/VACA1_triangle_LIN_XobsModel.csv", col_names = FALSE))
names <- c("Ours", "CNF", "DGP")  
custom_colors <- c("Ours" = "#1E88E5", "CNF" = "#FFC107", "DGP" = "red")  



Xmodel_df <- as.data.frame(Xmodel)
colnames(Xmodel_df) <- c("X1", "X2", "X3")
Xmodel_df$Type <- names[1]

Xref_df <- as.data.frame(Xref)
colnames(Xref_df) <- c("X1", "X2", "X3")
Xref_df$Type <- names[2]

XDGP_df <- as.data.frame(XDGP)
colnames(XDGP_df) <- c("X1", "X2", "X3")
XDGP_df$Type <- names[3]
all_data <- rbind(Xmodel_df, Xref_df, XDGP_df)

# Function to extract legend
get_legend <- function(my_plot){
  tmp <- ggplot2::ggplot_gtable(ggplot2::ggplot_build(my_plot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}


createPlotMatrix <- function(data, type_col, var_names,text_size = 20, axis_title_size = 18) {
  plot_list <- list()
  for (i in 1:3) {
    for (j in 1:3) {
      if (i == j) {
        p <- ggplot(data, aes_string(x = var_names[i], fill = type_col)) +
          geom_density(alpha = 0.4) +
          scale_fill_manual(values = custom_colors, name = "Methods") +
          theme_minimal() +
          theme(text = element_text(size = text_size), axis.title = element_text(size = axis_title_size)) +
          theme(legend.position = "none")
      } else if (i > j) {
        p <- ggplot(data, aes_string(x = var_names[j], y = var_names[i])) +
          geom_density_2d(aes_string(color = type_col), size = 0.5, breaks = c(0.01, 0.04)) +
          scale_color_manual(values = custom_colors, name = "Methods") +
          theme_minimal() +
          theme(text = element_text(size = text_size), axis.title = element_text(size = axis_title_size)) +
          theme(legend.position = "none")
      } else {
        sub_data <- data[sample(nrow(data), 5000), ]
        p <- ggplot(sub_data, aes_string(x = var_names[j], y = var_names[i], color = type_col)) +
          geom_point(shape = 1, alpha = 0.4) +
          scale_color_manual(values = custom_colors, name = "Methods") +
          theme_minimal() +
          theme(text = element_text(size = text_size), axis.title = element_text(size = axis_title_size)) +
          theme(legend.position = "none")
      }
      plot_list[[paste0(i, "_", j)]] = p
    }
  }
  
  # Combine plots using ggarrange function from ggpubr package
  combined <- ggarrange(
    plotlist = plot_list, 
    ncol = 3, nrow = 3, 
    common.legend = TRUE, 
    legend = "bottom"
  )
  
  return(combined)
}

# Sample function call
#library(cowplot)
g = createPlotMatrix(all_data, "Type", c("X1", "X2", "X3"), text_size = 18*1.5, axis_title_size=18*1.5)
g
ggsave(make_fn("observations.pdf"))
#ggsave(make_fn("observations_NSF.pdf"))
if (FALSE){
  file.copy(make_fn("observations.pdf"), '~/Dropbox/Apps/Overleaf/tramdag/figures/', overwrite = TRUE)
  file.copy(make_fn("observations_NSF.pdf"), '~/Dropbox/Apps/Overleaf/tramdag/figures/', overwrite = TRUE)
}

####### L2 Do Interventions on X2 #####################
dox_origs = c(-3,-1, 0)
num_samples = 25142L

#### Sampling for model and DGP
inter_mean_dgp_x2 = inter_mean_dgp_x3 = inter_mean_ours_x2 = inter_mean_ours_x3 = NA*dox_origs
inter_dgp_x2 = inter_dgp_x3 = inter_ours_x2 = inter_ours_x3 = matrix(NA, nrow=length(dox_origs), ncol=num_samples)
for (i in 1:length(dox_origs)){
  ### Our Model
  dox_orig = dox_origs[i]
  dox=scale_value(train$df_orig, col=2L, dox_orig) #On X2
  dat_do_x_s = do(thetaNN_l, train$A, doX = c(NA,dox,NA), num_samples = num_samples)
  
  df = unscale(train$df_orig, dat_do_x_s)
  inter_ours_x2[i,] = df$numpy()[,2]
  inter_ours_x3[i,] = df$numpy()[,3]

  ### DGP
  d = dgp(num_samples,doX2=dox_orig)
  inter_dgp_x2[i,] = d$df_orig[,2]$numpy()
  inter_dgp_x3[i,] = d$df_orig[,3]$numpy()
}

#### Reformating for ggplot
#Preparing a df for ggplot for selected do-values
df_do = data.frame(dox=numeric(0),x2=numeric(0),x3=numeric(0), type=character(0))
for (step in 1:length(dox_origs)){
  df_do = rbind(df_do, data.frame(
    dox = dox_origs[step],
    x2 = inter_dgp_x2[step,],
    x3 = inter_dgp_x3[step,],
    type = 'DGP'
  ))
  df_do = rbind(df_do, data.frame(
    dox = dox_origs[step],
    x2 = inter_ours_x2[step,],
    x3 = inter_ours_x3[step,],
    type = 'Ours'
  )
  )
}

### Loading the data from VACA2
NSF = TRUE
NSF = FALSE
if (NSF){
  X_inter <- read_csv("data/VACA1_triangle_lin/NSF/vaca1_triangle_lin_Xinter_x2=-3.csv", col_names = FALSE)
} else{
  X_inter <- read_csv("data/VACA1_triangle_lin/25K/vaca1_triangle_lin_Xinter_x2=-3.csv", col_names = FALSE)
}
df_do = rbind(df_do, data.frame(
  dox = -3,
  x2 = X_inter$X2,
  x3 = X_inter$X3,
  type = 'CNF' 
))

if (NSF){
  X_inter <- read_csv("data/VACA1_triangle_lin/NSF/vaca1_triangle_lin_Xinter_x2=-1.csv", col_names = FALSE)
} else{
  X_inter <- read_csv("data/VACA1_triangle_lin/25K/vaca1_triangle_lin_Xinter_x2=-1.csv", col_names = FALSE)
}
df_do = rbind(df_do, data.frame(
  dox = -1,
  x2 = X_inter$X2,
  x3 = X_inter$X3,
  type = 'CNF' 
))

if (NSF){
  X_inter <- read_csv("data/VACA1_triangle_lin/NSF/vaca1_triangle_lin_Xinter_x2=0.csv", col_names = FALSE)
} else{
  X_inter <- read_csv("data/VACA1_triangle_lin/25K/vaca1_triangle_lin_Xinter_x2=0.csv", col_names = FALSE)
}

df_do = rbind(df_do, data.frame(
  dox = 0,
  x2 = X_inter$X2,
  x3 = X_inter$X3,
  type = 'CNF' 
))


text_size = 20
axis_title_size = 18
#geom_density(alpha = 0.4) +
#  scale_fill_manual(values = custom_colors, name = "Methods")

# Custom labeller function
custom_labeller <- function(variable, value) {
  return(paste("doX2 =", value))
}

# Your ggplot code
ggplot(df_do) + 
  geom_density(aes(x=x3, fill=type), alpha=0.4, adjust = 1.5) + 
  xlim(-7, 5) +
  ylab("p(x3|do(x2)") +
  scale_fill_manual(values = custom_colors, name = "Methods") +
  facet_grid(~dox, labeller = custom_labeller) +  # Apply custom labeller here
  facet_grid(~dox, labeller = custom_labeller) +
  theme_minimal() +
  theme(text = element_text(size = text_size),
        axis.title = element_text(size = axis_title_size)) +
  theme(axis.text.x = element_text(angle = 90))#, panel.spacing = unit(1, "lines"))


ggsave(make_fn("dox2_dist_x3.pdf"), width = 15/1.7, height = 6/1.7)
if (FALSE){
  file.copy(make_fn("dox2_dist_x3.pdf"), '~/Dropbox/Apps/Overleaf/tramdag/figures/', overwrite = TRUE)
}






