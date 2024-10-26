##### Mr Browns MAC ####
if (FALSE){
  reticulate::use_python("~/miniforge3/envs/r-tensorflow/bin/python3.8", required = TRUE)
  library(reticulate)
  reticulate::py_config()
}
# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  args <- c(3, 'cs')
  args <- c(1, 'ls')
}
F32 <- as.numeric(args[1])
M32 <- args[2]
print(paste("FS:", F32, "M32:", M32))


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
DIR = 'summerof24/runs/triangle_structured_mixed/run_small_net'
if (!dir.exists(DIR)) {
  dir.create(DIR, recursive = TRUE)
}
# Copy this file to the directory DIR
file.copy('summerof24/triangle_structured_mixed.R', file.path(DIR, 'triangle_structured_mixed.R'), overwrite=TRUE)

num_epochs <- 500
len_theta = 20 # Number of coefficients of the Bernstein polynomials
hidden_features_I = c(2,2,2,2) 
hidden_features_CS = c(2,2,2,2)

if (F32 == 1){
  FUN_NAME = 'DPGLinear'
  f <- function(x) -0.3 * x
} else if (F32 == 2){
  f = function(x) 2 * x**3 + x
  FUN_NAME = 'DPG2x3+x'
} else if (F32 == 3){
  f = function(x) 0.5*exp(x)
  FUN_NAME = 'DPG0.5exp'
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
fn = file.path(DIR, paste0('triangle_mixed_', FUN_NAME, '_', MODEL_NAME))
print(paste0("Starting experiment ", fn))
   
xs = seq(-1,1,0.1)

plot(xs, f(xs), sub=fn, xlab='x2', ylab='f(x2)', main='DGP influence of x2 on x3')

##### DGP ########
dgp <- function(n_obs, doX=c(NA, NA, NA)) {
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
      #x_2_dash = 0.42 * x_2 + 2 * X_1
      X_2 = 1/0.42 * (x_2_dash - 2 * X_1)
      X_2 = 1/5. * (x_2_dash - 0.4 * X_1) # 0.39450
      X_2 = 1/5. * (x_2_dash - 1.2 * X_1) 
      X_2 = 1/5. * (x_2_dash - 2 * X_1)  # 
      
      
    } else{
      X_2 = rep(doX[2], n_obs)
    }
    
    #hist(X_2)
    #ds = seq(-5,5,0.1)
    #plot(ds, dlogis(ds))
    
    if (is.na(doX[3])){
      # x3 is an ordinal variable with K = 4 levels x3_1, x3_2, x3_3, x3_4
      # h(x3 | x1, x2) = h0 + gamma_1 * x1 + gamma_2 * x2
      # h0(x3_1) = theta_1, h0(x_3_2) =  theta_2, h0(x_3_3) = theta_3 
      theta_k = c(-2, 0.42, 1.02)
      
      h = matrix(, nrow=n_obs, ncol=3)
      for (i in 1:n_obs){
        h[i,] = theta_k + 0.2 * X_1[i] + f(X_2[i]) #- 0.3 * X_2[i]
      }
      
      U3 = rlogis(n_obs)
      # chooses the correct X value if U3 is smaller than -2 that is level one if it's between -2 and 0.42 it's level two answer on
      x3 = rep(1, n_obs)
      x3[U3 > h[,1]] = 2
      x3[U3 > h[,2]] = 3
      x3[U3 > h[,3]] = 4
      x3 = ordered(x3, levels=1:4)
    } else{
      x3 = rep(doX[3], n_obs)
    }
   
    #hist(X_3)
    A <- matrix(c(0, 1, 1, 0,0,1,0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
    dat.orig =  data.frame(x1 = X_1, x2 = X_2, x3 = x3)
    dat.tf = tf$constant(as.matrix(dat.orig), dtype = 'float32')
    
    q1 = quantile(dat.orig[,1], probs = c(0.05, 0.95)) 
    q2 = quantile(dat.orig[,2], probs = c(0.05, 0.95))
    q3 = c(1, 4) #No Quantiles for ordinal data
    
    
    return(list(
      df_orig=dat.tf, 
      df_R = dat.orig,
      #min =  tf$reduce_min(dat.tf, axis=0L),
      #max =  tf$reduce_max(dat.tf, axis=0L),
      min = tf$constant(c(q1[1], q2[1], q3[1]), dtype = 'float32'),
      max = tf$constant(c(q1[2], q2[2], q3[2]), dtype = 'float32'),
      type = c('c', 'c', 'o'),
      A=A))
} 

train = dgp(40000)
test  = dgp(10000)
(global_min = train$min)
(global_max = train$max)
data_type = train$type


len_theta_max = len_theta
for (i in 1:nrow(MA)){ #Maximum number of coefficients (BS and Levels - 1 for the ordinal)
  if (train$type[i] == 'o'){
    len_theta_max = max(len_theta_max, nlevels(train$df_R[,i]) - 1)
  }
}
param_model = create_param_model(MA, hidden_features_I=hidden_features_I, len_theta=len_theta, hidden_features_CS=hidden_features_CS)
optimizer = optimizer_adam()
param_model$compile(optimizer, loss=struct_dag_loss)
param_model$evaluate(x = train$df_orig, y=train$df_orig, batch_size = 7L)


##### Training ####
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

#pdf(paste0('loss_',fn,'.pdf'))
epochs = length(train_loss)
plot(1:length(train_loss), train_loss, type='l', main='Normal Training (green is valid)')
lines(1:length(train_loss), val_loss, type = 'l', col = 'green')

# Last 50
diff = max(epochs - 100,0)
plot(diff:epochs, val_loss[diff:epochs], type = 'l', col = 'green', main='Last 50 epochs')
lines(diff:epochs, train_loss[diff:epochs], type='l')

# plot(1:epochs, ws[,1], type='l', main='Coef', ylim=c(-0.5, 3))#, ylim=c(0, 6))
# abline(h=2, col='green')
# lines(1:epochs, ws[,2], type='l', ylim=c(0, 3))
# abline(h=0.2, col='green')
# lines(1:epochs, ws[,3], type='l', ylim=c(0, 3))
# abline(h=-0.3, col='green')


ggplot(ws, aes(x=1:nrow(ws))) + 
  geom_line(aes(y=w12, color='x1 --> x2')) + 
  geom_line(aes(y=w13, color='x1 --> x3')) + 
  geom_line(aes(y=w23, color='x2 --> x3')) + 
  geom_hline(aes(yintercept=2, color='x1 --> x2'), linetype=2) +
  geom_hline(aes(yintercept=0.2, color='x1 --> x3'), linetype=2) +
  geom_hline(aes(yintercept=-0.3, color='x2 --> x3'), linetype=2) +
  #scale_color_manual(values=c('x1 --> x2'='skyblue', 'x1 --> x3='red', 'x2 --> x3'='darkgreen')) +
  labs(x='Epoch', y='Coefficients') +
  theme_minimal() +
  theme(legend.title = element_blank())  # Removes the legend title
  

###### Coefficient Plot for Paper #######
if (FALSE){
  p = ggplot(ws, aes(x=1:nrow(ws))) + 
    geom_line(aes(y=w12, color="beta12")) + 
    geom_line(aes(y=w13, color="beta13")) + 
    geom_line(aes(y=w23, color="beta23")) + 
    geom_hline(aes(yintercept=2, color="beta12"), linetype=2) +
    geom_hline(aes(yintercept=0.2, color="beta13"), linetype=2) +
    geom_hline(aes(yintercept=-0.3, color="beta23"), linetype=2) +
    scale_color_manual(
      values=c('beta12'='skyblue', 'beta13'='red', 'beta23'='darkgreen'),
      labels=c(expression(beta[12]), expression(beta[13]), expression(beta[23]))
    ) +
    labs(x='Epoch', y='Coefficients') +
    theme_minimal() +
    theme(
      legend.title = element_blank(),   # Removes the legend title
      legend.position = c(0.85, 0.17),  # Adjust this to position the legend inside the plot (lower-right)
      legend.background = element_rect(fill="white", colour="black")  # Optional: white background with border
    )
  
  p
  file_name <- paste0(fn, "_coef_epoch.pdf")
  if (FALSE){
    file_path <- file.path("~/Library/CloudStorage/Dropbox/Apps/Overleaf/tramdag/figures", basename(file_name))
    ggsave(file_path, plot = p, width = 8, height = 6/2)
  }
}

param_model$get_layer(name = "beta")$get_weights() * param_model$get_layer(name = "beta")$mask


##### Checking observational distribution ####
s = do_dag_struct(param_model, train$A, doX=c(NA, NA, NA), num_samples = 5000)
plot(table(train$df_R[,3])/sum(table(train$df_R[,3])), ylab='Probability ', 
     main='Black = Observations, Red samples from TRAM-DAG',
     xlab='X3')
table(train$df_R[,3])/sum(table(train$df_R[,3]))
points(as.numeric(table(s[,3]$numpy()))/5000, col='red', lty=2)
table(s[,3]$numpy())/5000

par(mfrow=c(1,3))
for (i in 1:2){
  hist(train$df_orig$numpy()[,i], freq=FALSE, 100,main=paste0("X",i, " red: ours, black: data"), xlab='samples')
  #hist(train$df_orig$numpy()[,i], freq=FALSE, 100,main=paste0("X_",i))
  lines(density(s[,i]$numpy()), col='red')
}
plot(table(train$df_R[,3])/sum(table(train$df_R[,3])), ylab='Probability ', 
     main='Black = Observations, Red samples from TRAM-DAG',
     xlab='X3')
table(train$df_R[,3])/sum(table(train$df_R[,3]))
points(as.numeric(table(s[,3]$numpy()))/5000, col='red', lty=2)
table(s[,3]$numpy())/5000
par(mfrow=c(1,1))

######### Simulation of do-interventions #####
doX=c(0.2, NA, NA)
dx0.2 = dgp(10000, doX=doX)
dx0.2$df_orig$numpy()[1:5,]


doX=c(0.7, NA, NA)
dx7 = dgp(10000, doX=doX)
#hist(dx0.2$df_orig$numpy()[,2], freq=FALSE,100)
mean(dx7$df_orig$numpy()[,2]) - mean(dx0.2$df_orig$numpy()[,2])  
mean(dx7$df_orig$numpy()[,3]) - mean(dx0.2$df_orig$numpy()[,3])  

s_dag = do_dag_struct(param_model, train$A, doX=c(0.2, NA, NA))
hist(dx0.2$df_orig$numpy()[,2], freq=FALSE, 50, main='X2 | Do(X1=0.2)', xlab='samples', 
     sub='Histogram from DGP with do. red:TRAM_DAG')
sample_dag_0.2 = s_dag[,2]$numpy()
lines(density(sample_dag_0.2), col='red', lw=2)
m_x2_do_x10.2 = median(sample_dag_0.2)

i = 3 
d = dx0.2$df_orig$numpy()[,i]
plot(table(d)/length(d), ylab='Probability ', 
     main='X3 | do(X1=0.2)',
     xlab='X3', ylim=c(0,0.6),  sub='Black DGP with do. red:TRAM_DAG')
points(as.numeric(table(s_dag[,3]$numpy()))/nrow(s_dag), col='red', lty=2)

###### Figure for paper ######
if (FALSE){
  doX=c(NA, NA, NA)
  s_obs_fitted = do_dag_struct(param_model, train$A, doX, num_samples = 5000)$numpy()
  dx1 = -1
  doX=c(dx1, NA, NA)
  s_do_fitted = do_dag_struct(param_model, train$A, doX=doX)$numpy()
  
  df = data.frame(vals=s_obs_fitted[,1], type='Model', X=1, L='L0')
  df = rbind(df, data.frame(vals=s_obs_fitted[,2], type='Model', X=2, L='L0'))
  df = rbind(df, data.frame(vals=s_obs_fitted[,3], type='Model', X=3, L='L0'))
  
  df = rbind(df, data.frame(vals=train$df_R[,1], type='DGP', X=1, L='L0'))
  df = rbind(df, data.frame(vals=train$df_R[,2], type='DGP', X=2, L='L0'))
  df = rbind(df, data.frame(vals=as.numeric(train$df_R[,3]), type='DGP', X=3, L='L0'))
  
  df = rbind(df, data.frame(vals=s_do_fitted[,1], type='Model', X=1, L='L1'))
  df = rbind(df, data.frame(vals=s_do_fitted[,2], type='Model', X=2, L='L1'))
  df = rbind(df, data.frame(vals=s_do_fitted[,3], type='Model', X=3, L='L1'))
  
  d = dgp(10000, doX=doX)$df_R
  df = rbind(df, data.frame(vals=d[,1], type='DGP', X=1, L='L1'))
  df = rbind(df, data.frame(vals=d[,2], type='DGP', X=2, L='L1'))
  df = rbind(df, data.frame(vals=as.numeric(d[,3]), type='DGP', X=3, L='L1'))

  p = ggplot() +
    # For X = 1 and X = 2, use position = "identity" (no dodging)
    geom_histogram(data = subset(df, X != 3), 
                   aes(x=vals, col=type, fill=type, y=..density..), 
                   position = "identity", alpha=0.4) +
    # For X = 3, use a bar plot for discrete data
    geom_bar(data = subset(df, X == 3), 
             aes(x=vals, y=..prop.. * 4,  col=type, fill=type), 
             position = "dodge", alpha=0.4, size = 0.5)+
    #limit between 0,1 but not removing the data
    coord_cartesian(ylim = c(0, 4)) +
    facet_grid(L ~ X, scales = 'free',
               labeller = as_labeller(c('1' = 'X1', '2' = 'X2', '3' = 'X3', 'L1' = paste0('Do X1=',dx1), 'L0' = 'Obs')))+ 
    labs(y = "Density / (Frequency × 4)", x='')  + # Update y-axis label
    theme_minimal() +
    theme(
      legend.title = element_blank(),   # Removes the legend title
      legend.position = c(0.17, 0.25),  # Adjust this to position the legend inside the plot (lower-right)
      legend.background = element_rect(fill="white", colour="white")  # Optional: white background with border
    )
  p
  file_name <- paste0(fn, "_L0_L1.pdf")
  ggsave(file_name, plot=p, width = 8, height = 6)
  if (FALSE){
    file_path <- file.path("~/Library/CloudStorage/Dropbox/Apps/Overleaf/tramdag/figures", basename(file_name))
    print(file_path)
    ggsave(file_path, plot=p, width = 8/2, height = 6/2)
  }
      
}




s_dag = do_dag_struct(param_model, train$A, doX=c(0.7, NA, NA))
i = 2
ds = dx7$df_orig$numpy()[,i]
hist(ds, freq=FALSE, 50, main='X2 | Do(X1=0.7)', xlab='samples', 
     sub='Histogram from DGP with do. red:TRAM_DAG')
sample_dag_07 = s_dag[,i]$numpy()
lines(density(sample_dag_07), col='red', lw=2)
m_x2_do_x10.7 = median(sample_dag_07)
m_x2_do_x10.7 - m_x2_do_x10.2

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
  shift_23[i] = param_model(X)[1,3,2]$numpy() #2-LS Term X2-->X3 (Ms. Whites Notation)
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
abline(0, .2)


if (F32 == 1){ #Linear DGP
  if (MA[2,3] == 'ls'){
    delta_0 = shift_23[idx0] - f(0)
    plot(xs, shift_23 - delta_0, main='LS-Term (black DGP, red Ours)', 
         sub = paste0('Effect of x2 on x3, delta_0 ', round(delta_0,2)),
         xlab='x2', col='red')
    #abline(shift_23[length(shift_23)/2], -0.3)
    abline(0, -0.3)
  } 
  if (MA[2,3] == 'cs'){
    plot(xs, cs_23, main='CS-Term (black DGP, red Ours)', xlab='x2',  
         sub = 'Effect of x2 on x3',col='red')
    
    abline(cs_23[idx0], -0.3)  
  }
} else{ #Non-Linear DGP
  if (MA[2,3] == 'ls'){
    delta_0 = shift_23[idx0] - f(0)
    plot(xs, shift_23 - delta_0, main='LS-Term (black DGP, red Ours)', 
         sub = paste0('Effect of x2 on x3, delta_0 ', round(delta_0,2)),
         xlab='x2', col='red')
    lines(xs, f(xs))
  } else if (MA[2,3] == 'cs'){
    plot(xs, cs_23 + ( -cs_23[idx0] + f(0) ),
         ylab='CS',
         main='CS-Term (black DGP f2(x), red Ours)', xlab='x2',  
         sub = 'Effect of x2 on x3',col='red')
    lines(xs, f(xs))
  } else{
    print(paste0("Unknown Model ", MA[2,3]))
  }
}
#plot(xs,f(xs), xlab='x2', main='DGP')
par(mfrow=c(1,1))


if (FALSE){
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

h_12 = h_I + h_LS[,cont_dims, drop=FALSE] + h_CS[,cont_dims, drop=FALSE]

### Ordingal Dimensions
B = tf$shape(t_i)[1]
col = 3
nol = tf$cast(k_max[col] - 1L, tf$int32) # Number of cut-points in respective dimension
theta_ord = theta[,col,1:nol,drop=TRUE] # Intercept
h_3 = theta_ord + h_LS[,col, drop=FALSE] + h_CS[,col, drop=FALSE]

####### DGP Transformations #######
X_1 = t_i[,1]$numpy()
X_2 = t_i[,2]$numpy()
h2_DGP = 5 *X_2 + 2 * X_1
plot(h2_DGP[1:2000], h_12[1:2000,2]$numpy())
abline(0,1,col='red')

h2_DGP_I = 5*X_2
h2_M_I = h_I[,2]

plot(h2_DGP_I, h2_M_I)
abline(0,1,col='red')

h_3 #Model

##### DGP 
theta_k = c(-2, 0.42, 1.02)
n_obs = B$numpy()
h_3_DPG = matrix(, nrow=n_obs, ncol=3)
for (i in 1:n_obs){
  h_3_DPG[i,] = theta_k + 0.2 * X_1[i] + f(X_2[i]) #- 0.3 * X_2[i]
}

plot(h_3_DPG[1:2000,3], h_3[1:2000,3]$numpy())
abline(0,1,col='green')

#LS
plot(-0.2*X_1, h_LS[,3]$numpy())
abline(0,1,col='green')

#LS
plot(f(X_2), h_CS[,3]$numpy())
abline(0,1,col='green')
}






