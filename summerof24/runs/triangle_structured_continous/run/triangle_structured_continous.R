##### Mr Browns MAC ####
reticulate::use_python("~/miniforge3/envs/r-tensorflow/bin/python3.8", required = TRUE)
library(reticulate)
reticulate::py_config()

# Get command-line arguments - if called via sh
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {  # if not called via sh
  args <- c(1, 'ls') #
  args <- c(3, 'cs') # For figure XXX in the paper
}
F32 <- as.numeric(args[1])
M32 <- args[2]
print(paste("F32:", F32, "M32:", M32))


#### A mixture of discrete and continuous variables ####
library(tensorflow)
library(keras)
library(mlt)
library(tram)
library(MASS)
library(tensorflow)
library(keras)
library(tidyverse)
source('summerof24/utils_tf.R')

#### For TFP
library(tfprobability)
source('summerof24/utils_tfp.R')

##### Flavor of experiment ######

#### Saving the current version of the script into runtime
DIR = 'summerof24/runs/triangle_structured_continous/run_nodes25'
if (!dir.exists(DIR)) {
  dir.create(DIR, recursive = TRUE)
}
# Copy this file to the directory DIR
file.copy('summerof24/triangle_structured_continous.R', file.path(DIR, 'triangle_structured_continous.R'), overwrite=TRUE)

num_epochs <- 500
len_theta = 20 # Number of coefficients of the Bernstein polynomials
hidden_features_I = c(2,25,25,2)    #hidden_features_CS=hidden_features_I = c(2,25,25,2)
hidden_features_CS = c(2,25,25,2)

SEED = -1 #If seed > 0 then the seed is set

if (F32 == 1){
  FUN_NAME = 'DPGLinear'
  f <- function(x) -0.3 * x
} else if (F32 == 2){
  f = function(x) 2 * x**3 + x
  FUN_NAME = 'DPG2x3+x'
} else if (F32 == 3){
  f = function(x) 0.5*exp(x)
  FUN_NAME = 'DPG0.5exp'
} else if (F32 == 4){
  f = function(x) 0.75*atan(5*(x+0.12)) 
  FUN_NAME = 'DPGatan'
} else if (F32 == 5){
  f = function(x) 2*sin(3*x)+x 
  FUN_NAME = 'DPGSin'
} else {
  stop("Unknown Function F32")
}

# xs = seq(-1,1,0.1)
# f = function(x) 2 * x**3 + 0.1*(x-.5)**5
if (FALSE){
  f = function(x) 0.75*atan(5*(x+0.12)) 
  plot(xs, f(xs))
  s=train$df_R$x2
  hist(s, freq=FALSE, 100)
  hist(f(s), freq=FALSE, 100)
}

if (M32 == 'ls') {
  MA =  matrix(c(
    0, 'ls', 'ls', 
    0,    0, 'ls', 
    0,    0,   0), nrow = 3, ncol = 3, byrow = TRUE)
  MODEL_NAME = 'ModelLS'
} else{
  MA =  matrix(c(
    0, 'ls', 'ls', 
    0,    0, 'cs', 
    0,    0,   0), nrow = 3, ncol = 3, byrow = TRUE)
  MODEL_NAME = 'ModelCS'
}


# fn = 'triangle_mixed_DGPLinear_ModelLinear.h5'
# fn = 'triangle_mixed_DGPSin_ModelCS.h5'

if (SEED < 0){
  fn = file.path(DIR, paste0('triangle_mixed_', FUN_NAME, '_', MODEL_NAME))
} else{
  fn = file.path(DIR, paste0('triangle_mixed_', FUN_NAME, '_', MODEL_NAME, '_SEED', SEED))
}
print(paste0("Starting experiment ", fn))
   
xs = seq(-1,1,0.1)

plot(xs, -f(xs), sub=fn, xlab='x2', ylab='f(x2)', main='DGP influence of x2 on x3', cex.sub=0.4)
##### DGP ########
dgp <- function(n_obs, doX=c(NA, NA, NA), seed=-1) {
    if (seed > 0) {
      set.seed(seed)
      print(paste0("Setting Seed:", seed))
    }
    #n_obs = 1e5 n_obs = 10
    #Sample X_1 from GMM with 2 components
    if (is.na(doX[1])){
      X_1_A = rnorm(n_obs, 0.25, 0.1)
      X_1_B = rnorm(n_obs, 0.73, 0.05)
      X_1 = ifelse(sample(1:2, replace = TRUE, size = n_obs) == 1, X_1_A, X_1_B)
    } else{
      X_1 = rep(doX[1], n_obs)
    }
    #hist(X_1)
    
    # Sampling according to colr
    if (is.na(doX[2])){
      U2 = runif(n_obs)
      
      x_2_dash = qlogis(U2)
      #x_2_dash = h_0(x_2) + beta * X_1
      X_2 = 1/0.42 * (x_2_dash - 2 * X_1)
      X_2 = 1/5. * (x_2_dash - 0.4 * X_1) # 0.39450
      X_2 = 1/5. * (x_2_dash - 1.2 * X_1) 
      #h2 = x_2_dash = 5 * x_2 + 2 * X_1
      X_2 = 1/5. * (x_2_dash - 2 * X_1)  # 
      
      
    } else{
      X_2 = rep(doX[2], n_obs)
    }
    
    #hist(X_2)
    #ds = seq(-5,5,0.1)
    #plot(ds, dlogis(ds))
    if (is.na(doX[3])){
      U3 = runif(n_obs)
      x_3_dash = qlogis(U3)
      #h(x3|x1,x2) = 0.63*x3 - 0.2*x1 - f(x2)
      #x_3_dash = h_0_3(x_3) + gamma_1 * X_1 + gamma_2 * X_2
      #x_3_dash = 0.63 * x_3 -0.2 * X_1 + 1.3 * X_2
      #x_3_dash = h(x3|x1,x2) = 0.63*x3 - 0.2*x1 - f(x2)
      X_3 = (x_3_dash + 0.2 * X_1 + f(X_2))/0.63
    } else{
      X_3 = rep(doX[3], n_obs)
    }
    
   
    #hist(X_3)
    A <- matrix(c(0, 1, 1, 0,0,1,0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
    dat.orig =  data.frame(x1 = X_1, x2 = X_2, x3 = X_3)
    dat.tf = tf$constant(as.matrix(dat.orig), dtype = 'float32')
    
    q1 = quantile(dat.orig[,1], probs = c(0.05, 0.95)) 
    q2 = quantile(dat.orig[,2], probs = c(0.05, 0.95))
    q3 = quantile(dat.orig[,3], probs = c(0.05, 0.95))
    
    
    return(list(
      df_orig=dat.tf, 
      df_R = dat.orig,
      #min =  tf$reduce_min(dat.tf, axis=0L),
      #max =  tf$reduce_max(dat.tf, axis=0L),
      min = tf$constant(c(q1[1], q2[1], q3[1]), dtype = 'float32'),
      max = tf$constant(c(q1[2], q2[2], q3[2]), dtype = 'float32'),
      type = c('c', 'c', 'c'),
      A=A))
} 

train = dgp(40000, seed=ifelse(SEED > 0, SEED, -1))
test  = dgp(40000, seed=ifelse(SEED > 0, SEED + 1, -1))
(global_min = train$min)
(global_max = train$max)
data_type = train$type



len_theta_max = len_theta
for (i in 1:nrow(MA)){ #Maximum number of coefficients (BS and Levels - 1 for the ordinal)
  if (train$type[i] == 'o'){
    len_theta_max = max(len_theta_max, nlevels(train$df_R[,i]) - 1)
  }
}
param_model = create_param_model(MA, hidden_features_I = hidden_features_I, 
                                 len_theta = len_theta_max, 
                                 hidden_features_CS = hidden_features_CS)

h_params = param_model(train$df_orig)
# loss before training
struct_dag_loss(t_i=train$df_orig, h_params=h_params)
param_model = create_param_model(MA, hidden_features_I=hidden_features_I, len_theta=len_theta, hidden_features_CS=hidden_features_CS)
optimizer = optimizer_adam()
param_model$compile(optimizer, loss=struct_dag_loss)
param_model$evaluate(x = train$df_orig, y=train$df_orig, batch_size = 7L)


##### Training or readin of weights if h5 available ####
fnh5 = paste0(fn, '_E', num_epochs, '.h5')
fnRdata = paste0(fn, '_E', num_epochs, '.RData')
if (file.exists(fnh5)){
  param_model$load_weights(fnh5)
  load(fnRdata) #Loading of the workspace causes trouble e.g. param_model is zero
  # Quick Fix since loading global_min causes problem (no tensors as RDS)
  (global_min = train$min)
  (global_max = train$max)
} else {
  if (FALSE){ ### Full Training w/o diagnostics
    hist = param_model$fit(x = train$df_orig, y=train$df_orig, epochs = 200L,verbose = TRUE)
    param_model$save_weights(fn)
    plot(hist$epoch, hist$history$loss)
    plot(hist$epoch, hist$history$loss, ylim=c(1.07, 1.2))
  } else { ### Training with diagnostics
    ws <- data.frame(w12 = numeric())
    train_loss <- numeric()
    val_loss <- numeric()
    
    # Training loop
    for (e in 1:num_epochs) {
      print(paste("Epoch", e))
      hist <- param_model$fit(x = train$df_orig, y = train$df_orig, 
                              epochs = 1L, verbose = TRUE, 
                              validation_data = list(test$df_orig,test$df_orig))
      
      # Append losses to history
      train_loss <- c(train_loss, hist$history$loss)
      val_loss <- c(val_loss, hist$history$val_loss)
      
      # Extract specific weights
      w <- param_model$get_layer(name = "beta")$get_weights()[[1]]
      
      ws <- rbind(ws, data.frame(w12 = w[1, 2], w13 = w[1, 3], w23 = w[2, 3]))
    }
    # Save the model
    param_model$save_weights(fnh5)
    save(train_loss, val_loss, train_loss, f, MA, len_theta,
         hidden_features_I,
         hidden_features_CS,
         ws,
         #global_min, global_max,
         file = fnRdata)
  }
}

####### FINISHED TRAINING #####
#pdf(paste0('loss_',fn,'.pdf'))
epochs = length(train_loss)
plot(1:length(train_loss), train_loss, type='l', main='Training (black: train, green: valid)')
lines(1:length(train_loss), val_loss, type = 'l', col = 'green')

# Last 50
diff = max(epochs - 50,1)
plot(diff:epochs, val_loss[diff:epochs], type = 'l', col = 'green', main='Last 50 epochs')
lines(diff:epochs, train_loss[diff:epochs], type='l')


#### Plotting of the Loss Curve ##########
tail(ws)
if (FALSE){
  p = ggplot(ws, aes(x=1:nrow(ws))) + 
    geom_line(aes(y=w12, color="beta12")) + 
    geom_line(aes(y=w13, color="beta13")) + 
    geom_line(aes(y=w23, color="beta23")) + 
    geom_hline(aes(yintercept=2, color="beta12"), linetype=2) +
    geom_hline(aes(yintercept=-0.2, color="beta13"), linetype=2) +
    geom_hline(aes(yintercept=+0.3, color="beta23"), linetype=2) +
    scale_color_manual(
      values=c('beta12'='skyblue', 'beta13'='red', 'beta23'='darkgreen'),
      labels=c(expression(beta[12]), expression(beta[13]), expression(beta[23]))
    ) +
    labs(x='Epoch', y='Coefficients') +
    theme_minimal() +
    theme(
      legend.title = element_blank(),   # Removes the legend title
      legend.position = c(0.85, 0.25),  # Adjust this to position the legend inside the plot (lower-right)
      legend.background = element_rect(fill="white", colour="black")  # Optional: white background with border
    )
  if (FALSE){
   ### NOTE THAT WE RENAMED THE PRODUCED FILE and added the PATH due to a naming conflict (the continuous files are wrongly named mixed)
    # Remove 'mixed' in filename
    file_name <- gsub("mixed", "", file_name)
    file_path <- file.path("~/Library/CloudStorage/Dropbox/Apps/Overleaf/tramdag/figures", basename(file_name))
    ggsave(file_path, plot = p, width = 8, height = 6/2)  
  }
}

if (FALSE){
# Creating the figure for the paper 
# triangle_mixed_DPGLinear_ModelLS_coef_epoch 
  p = ggplot(ws, aes(x=1:nrow(ws))) + 
    geom_line(aes(y=w12, color="beta12")) + 
    geom_line(aes(y=w13, color="beta13")) + 
    geom_line(aes(y=w23, color="beta23")) + 
    geom_hline(aes(yintercept=2, color="beta12"), linetype=2) +
    geom_hline(aes(yintercept=-0.2, color="beta13"), linetype=2) +
    geom_hline(aes(yintercept=+0.3, color="beta23"), linetype=2) +
    scale_color_manual(
      values=c('beta12'='skyblue', 'beta13'='red', 'beta23'='darkgreen'),
      labels=c(expression(beta[12]), expression(beta[13]), expression(beta[23]))
    ) +
    labs(x='Epoch', y='Coefficients') +
    theme_minimal() +
    theme(
      legend.title = element_blank(),   # Removes the legend title
      legend.position = c(0.85, 0.25),  # Adjust this to position the legend inside the plot (lower-right)
      legend.background = element_rect(fill="white", colour="black")  # Optional: white background with border
    )
  
  file_name <- paste0(fn, "_coef_epoch.pdf")
  # Save the plot
  ggsave(file_name, plot = p, width = 8, height = 6)
  file_path <- file.path("~/Library/CloudStorage/Dropbox/Apps/Overleaf/tramdag/figures", basename(file_name))
  ggsave(file_path, plot = p, width = 8/2, height = 6/2)
}

param_model$evaluate(x = train$df_orig, y=train$df_scaled) #Does not work, probably TF Eager vs Compiled
# One more step to estimate NLL
if (FALSE){
  vals = NULL
  for (i in 1:10){
    test  = dgp(40000, i+10001)
    hist = param_model$fit(x = train$df_orig, y = train$df_orig, 
                    epochs = 1L, verbose = TRUE, 
                    validation_data = list(test$df_orig,test$df_orig))
    vals = append(vals, hist$history$val_loss)
  }
  t.test(vals)
  M32
  F32
}
fn
len_theta
param_model$get_layer(name = "beta")$get_weights() * param_model$get_layer(name = "beta")$mask


#### Checking the transformation ####
h_params = param_model(train$df_orig)
r = check_baselinetrafo(h_params)
Xs = r$Xs
h_I = r$h_I

##### X1

df = data.frame(train$df_orig$numpy())
fit.21 = Colr(X2~X1,df, order=len_theta)
temp = model.frame(fit.21)[1:2,-1, drop=FALSE] #WTF!
plot(fit.21, which = 'baseline only', newdata = temp, lwd=2, col='blue', 
     main='h_I(X2) Black: COLR, Red: Our Model', cex.main=0.8)
lines(Xs[,2], h_I[,2], col='red', lty=2, lwd=5)
rug(train$df_orig$numpy()[,2], col='blue')

fit.312 = Colr(X3 ~ X1 + X2,df, order=len_theta)
temp = model.frame(fit.312)[1:2, -1, drop=FALSE] #WTF!

plot(fit.312, which = 'baseline only', newdata = temp, lwd=2, col='blue', 
     main='h_I(X3) Colr and Our Model', cex.main=0.8)
lines(Xs[,3], h_I[,3], col='red', lty=2, lwd=5)
rug(train$df_orig$numpy()[,3], col='blue')


##### Checking observational distribution ####
library(car)
s = do_dag_struct(param_model, train$A, doX=c(NA, NA, NA), num_samples = 5000)
par(mfrow=c(1,3))
for (i in 1:3){
  d = s[,i]$numpy()
  hist(train$df_orig$numpy()[,i], freq=FALSE, 100,main=paste0("X",i, " red: ours, black: data"), xlab='samples')
  lines(density(train$df_orig$numpy()[,i]), col='blue')
  #hist(train$df_orig$numpy()[,i], freq=FALSE, 100,main=paste0("X_",i))
  lines(density(s[,i]$numpy()), col='red')
  #qqplot(train$df_orig$numpy()[,i], s[,i]$numpy())
  #abline(0,1)
}
par(mfrow=c(1,1))

######### Simulation of do-interventions #####
doX=c(0.2, NA, NA)
dx0.2 = dgp(10000, doX=doX, seed=SEED)
dx0.2$df_orig$numpy()[1:5,]


doX=c(0.7, NA, NA)
dx7 = dgp(10000, doX=doX, seed=SEED)
#hist(dx0.2$df_orig$numpy()[,2], freq=FALSE,100)
mean(dx7$df_orig$numpy()[,2]) - mean(dx0.2$df_orig$numpy()[,2])  
mean(dx7$df_orig$numpy()[,3]) - mean(dx0.2$df_orig$numpy()[,3])  

s_dag = do_dag_struct(param_model, train$A, doX=c(0.2, NA, NA))
hist(dx0.2$df_orig$numpy()[,2], freq=FALSE, 50, main='X2 | Do(X1=0.2)', xlab='samples', 
     sub='Histogram from DGP with do. red:TRAM_DAG')
sample_dag_0.2 = s_dag[,2]$numpy()
lines(density(sample_dag_0.2), col='red', lw=2)
m_x2_do_x10.2 = median(sample_dag_0.2)

doX=c(NA, -1, NA)
s_dag = do_dag_struct(param_model, train$A, doX)
sdgp = dgp(10000, doX=doX, seed=SEED)
hist(sdgp$df_orig$numpy()[,3], freq=FALSE, 50, xlab='samples', 
     sub='Histogram from DGP with do. red:TRAM_DAG')
sample_dag = s_dag[,3]$numpy()
lines(density(sample_dag), col='red', lw=2)

doX=c(1, NA, NA)
s_dag = do_dag_struct(param_model, train$A, doX)
sdgp = dgp(10000, doX=doX, seed=SEED)
hist(sdgp$df_orig$numpy()[,2], freq=FALSE, 50, xlab='samples', 
     sub='Histogram from DGP with do. red:TRAM_DAG')
sample_dag = s_dag[,2]$numpy()
lines(density(sample_dag), col='red', lw=2)

###### Comparison of estimated f(x2) vs TRUE f(x2) #######
shift_12 = shift_23 = shift1 = cs_23 = xs = seq(-1,1,length.out=41)
idx0 = which(xs == 0) #Index of 0 xs needs to be odd
for (i in 1:length(xs)){
  #i = 1
  x = xs[i]
  # Varying x1
  X = tf$constant(c(x, 0.5, 3), shape=c(1L,3L)) 
  shift1[i] =   param_model(X)[1,3,2]$numpy() #2=LS Term X1->X3
  shift_12[i] = param_model(X)[1,2,2]$numpy() #2=LS Term X1->X2
  
  #Varying x2
  X = tf$constant(c(0.5, x, 3), shape=c(1L,3L)) 
  cs_23[i] = param_model(X)[1,3,1]$numpy() #1=CS Term
  shift_23[i] = param_model(X)[1,3,2]$numpy() #2-LS Term X2-->X3 (Mrs. Whites' Notation)
}

######### Learned Transformation of f(x2) ########
if (FALSE){
  if (MA[2,3] == 'cs' && F32 == 1){
    # Assuming xs, cs_23, and idx0 are predefined vectors
    # Create a data frame for the ggplot
    df <- data.frame(x2 = xs, cs_23 = cs_23)
    
    # Create the ggplot
    p <- ggplot(df, aes(x = x2, y = cs_23)) +
      geom_line(aes(color = "Complex Shift Estimate"), size = 1) +  
      geom_point(aes(color = "Complex Shift Estimate"), size = 1) + 
      geom_abline(aes(color = "f"), intercept = cs_23[idx0], slope = 0.3, size = 1) +  # Black solid line for 'DGP'
      scale_color_manual(
        values = c("Complex Shift Estimate" = "blue", "f" = "black"),  # Set colors
        labels = c("Complex Shift Estimate", "f(x)")  # Custom legend labels with expression for f(X_2)
      ) +
      labs(
        x = expression(x[2]),  # Subscript for x_2
        y = paste("~f(x2)"),  # Optionally leave y-axis label blank
        color = NULL  # Removes the color legend title
      ) +
      theme_minimal() +
      theme(legend.position = "none")  # Correct way to remove the legend
    
    # Display the plot
    p
  } else if (MA[2,3] == 'cs' && F32 != 1){
    # Assuming xs, shift_23, and idx0 are predefined vectors
    # Create a data frame for the ggplot
    df <- data.frame(x2 = xs, 
                     shift_23 = cs_23 + ( -cs_23[idx0] - f(0)),
                     f = -f(xs)
                     )
    # Create the ggplot
    p <- ggplot(df, aes(x = x2, y = shift_23)) +
      #geom_line(aes(color = "Shift Estimate"), size = 1) +  # Blue line for 'Shift Estimate'
      geom_point(aes(color = "Shift Estimate"), size = 1) +  # Blue points for 'Shift Estimate'
      geom_line(aes(color = "f", y = f), ) +  # Black solid line for 'DGP'
      scale_color_manual(
        values = c("Shift Estimate" = "blue", "f" = "black"),  # Set colors
        labels = c("Shift Estimate", "f(x2)")  # Custom legend labels with expression for f(X_2)
      ) +
      labs(
        x = expression(x[2]),  # Subscript for x_2
        y = "~f(x2)",  # Optionally leave y-axis label blank
        color = NULL  # Removes the color legend title
      ) +
      theme_minimal() +
      theme(legend.position = "none")  # Correct way to remove the legend
    
    # Display the plot
    p
  } else{
    print(paste0("Unknown Model ", MA[2,3]))
  }
 
  file_name <- paste0(fn, "_f23_est.pdf")
  # Save the plot
  ggsave(file_name, plot = p, width = 8, height = 8)
  file_path <- file.path("~/Library/CloudStorage/Dropbox/Apps/Overleaf/tramdag/figures", basename(file_name))
  ggsave(file_path, plot = p, width = 8/3, height = 8/3)
}
    

par(mfrow=c(2,2))
plot(xs, shift_12, main='LS-Term (black DGP, red Ours)', 
     sub = 'Effect of x1 on x2',
     xlab='x1', col='red')
abline(0, 2)

delta_0 = shift1[idx0] - 0
plot(xs, shift1 - delta_0, main='LS-Term (black DGP, red Ours)', 
     sub = paste0('Effect of x1 on x3, delta_0 ', round(delta_0,2)),
     xlab='x1', col='red')
abline(0, -.2)


if (F32 == 1){ #Linear DGP
  if (MA[2,3] == 'ls'){
    delta_0 = shift_23[idx0] - f(0)
    plot(xs, shift_23 - delta_0, main='LS-Term (black DGP, red Ours)', 
         sub = paste0('Effect of x2 on x3, delta_0 ', round(delta_0,2)),
         xlab='x2', col='red')
    #abline(shift_23[length(shift_23)/2], -0.3)
    abline(0, 0.3)
  } 
  if (MA[2,3] == 'cs'){
    plot(xs, cs_23, main='CS-Term (black DGP, red Ours)', xlab='x2',  
         sub = 'Effect of x2 on x3',col='red')
    
    abline(cs_23[idx0], 0.3)  
  }
} else{ #Non-Linear DGP
  if (MA[2,3] == 'ls'){
    delta_0 = shift_23[idx0] + f(0)
    plot(xs, shift_23 - delta_0, main='LS-Term (black DGP, red Ours)', 
         sub = paste0('Effect of x2 on x3, delta_0 ', round(delta_0,2)),
         xlab='x2', col='red')
    lines(xs, -f(xs))
  } else if (MA[2,3] == 'cs'){
    plot(xs, cs_23 + ( -cs_23[idx0] - f(0) ),
         ylab='CS',
         main='CS-Term (black DGP f2(x), red Ours)', xlab='x2',  
         sub = 'Effect of x2 on x3',col='red')
    lines(xs, -f(xs))
  } else{
    print(paste0("Unknown Model ", MA[2,3]))
  }
}
#plot(xs,f(xs), xlab='x2', main='DGP')
par(mfrow=c(1,1))


if (TRUE){
####### Compplete transformation Function #######
### Copied from structured DAG Loss
t_i = train$df_orig
k_min <- k_constant(global_min)
k_max <- k_constant(global_max)

# from the last dimension of h_params the first entriy is h_cs1
# the second to |X|+1 are the LS
# the 2+|X|+1 to the end is H_I
h_cs <- h_params[,,1, drop = FALSE]
h_ls <- h_params[,,2, drop = FALSE]
#LS
h_LS = tf$squeeze(h_ls, axis=-1L)#tf$einsum('bx,bxx->bx', t_i, beta)
#CS
h_CS = tf$squeeze(h_cs, axis=-1L)

theta_tilde <- h_params[,,3:dim(h_params)[3], drop = FALSE]
theta = to_theta3(theta_tilde)
cont_dims = which(data_type == 'c') #1 2
cont_ord = which(data_type == 'o') #3

### Continiuous dimensions
#### At least one continuous dimension exits
h_I = h_dag_extra(t_i[,cont_dims, drop=FALSE], theta[,cont_dims,1:len_theta,drop=FALSE], k_min[cont_dims], k_max[cont_dims]) 

h = h_I + h_LS[,cont_dims, drop=FALSE] + h_CS[,cont_dims, drop=FALSE]

####### DGP Transformations #######
X_1 = t_i[,1]$numpy()
X_2 = t_i[,2]$numpy()
X_3 = t_i[,3]$numpy()

#h2 = x_2_dash = 5 * x_2 + 2 * X_1
h2_DGP = 5 *X_2 + 2 * X_1
h2_DGP_LS = 2 * X_1
h2_DGP_CS = rep(0, length(X_2))
h2_DGP_I = 5 * X_2

#h(x3|x1,x2) = 0.63*x3 - 0.2*x1 - f(x2)
h3_DGP = 0.63*X_3 - 0.2*X_1 - f(X_2)
h3_DGP_LS = -0.2*X_1
h3_DGP_CS = -f(X_2)
h3_DGP_I = 0.63*X_3


par(mfrow=c(2,2))
plot(h2_DGP, h[,2]$numpy(), main='h2')
abline(0,1,col='red')
confint(lm(h[,2]$numpy() ~ h2_DGP))

#Same for Intercept
plot(h2_DGP_I, h_I[,2]$numpy(), main='h2_I')
abline(0,1,col='red')
confint(lm(h_I[,2]$numpy() ~ h2_DGP_I))

plot(h2_DGP_LS, h_LS[,2]$numpy(), main='h2_LS')
abline(0,1,col='red')
confint(lm(h_LS[,2]$numpy() ~ h2_DGP_LS))

#Same for CS
plot(h2_DGP_CS, h_CS[,2]$numpy(), main='h2_CS')
abline(0,1,col='red')
confint(lm(h_CS[,2]$numpy() ~ h2_DGP_CS))

par(mfrow=c(1,1))


par(mfrow=c(2,2))

plot(h3_DGP, h[,3]$numpy(), main='h3')
abline(0,1,col='red')
confint(lm(h[,3]$numpy() ~ h3_DGP))

plot(h3_DGP_I, h_I[,3]$numpy(), main='h3_I')
abline(0,1,col='red')
confint(lm(h_I[,3]$numpy() ~ h3_DGP_I))

#same for ls  
plot(h3_DGP_LS, h_LS[,3]$numpy(), main='h3_LS')
abline(0,1,col='red')
confint(lm(h_LS[,3]$numpy() ~ h3_DGP_LS))

#same for CS
plot(h3_DGP_CS, h_CS[,3]$numpy(), main='h3_CS')
abline(0,1,col='red')
confint(lm(h_CS[,3]$numpy() ~ h3_DGP_CS))

par(mfrow=c(1,1))

}






