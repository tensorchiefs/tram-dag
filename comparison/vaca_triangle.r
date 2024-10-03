##########
# Code for fitting vaca_triangle_model
# This is sourced in Figure_Triangle_Linear_Bimodal.R and Figure_Triangle_NLLinear.R (TOBE_DONE)

#######################
# Latent Distribution
latent_dist = tfd_logistic(loc=0, scale=1)
#latent_dist = tfd_normal(loc=0, scale=1)
#latent_dist = tfd_truncated_normal(loc=0., scale=1.,low=-4,high = 4)


len_theta = M + 1
bp = make_bernp(len_theta)

######################################
############# DGP ###############
######################################
dgp <- function(n_obs, coeffs = NULL, doX1=NA, doX2=NA, dat_train=NULL, seed=NA, file=NULL) {
  if (is.na(seed) == FALSE){
    set.seed(seed)
  }
  
  #Use external data 
  if (is.null(file) == FALSE){
    stop("Not implemented yet")
    #data <- read.csv(file, header = FALSE)
    #X_1 <- data[,1]
    #...
    #X_4 <- data[,4]
    #n_obs=length(X_4)
  } else{
    if (VACA == 2 && TYPE == 'NONLIN'){
      print("=== Using the DGP of the VACA2 paper in the non-linear Fashion")
      X_1 = 1 + rnorm(n_obs)
      if (is.na(doX1) == FALSE){
        X_1 = X_1 * 0 + doX1
      }
      X_2 = 2*X_1^2 + rnorm(n_obs)
      if (is.na(doX2) == FALSE){
        X_2 = X_2 * 0 + doX2
      }
      X_3 = 20./(1 + exp(-X_2^2 + X_1)) + rnorm(n_obs)
    } else if (VACA == 1 && TYPE == 'LIN'){
      print("=== Using the DGP of the VACA1 paper in the linear Fashion (Tables 5/6)")
      flip = sample(c(0,1), n_obs, replace = TRUE)
      X_1 = flip*rnorm(n_obs, -2, sqrt(1.5)) + (1-flip) * rnorm(n_obs, 1.5, 1)
      if (is.na(doX1) == FALSE){
        X_1 = X_1 * 0 + doX1
      }
      X_2 = -X_1 + rnorm(n_obs)
      if (is.na(doX2) == FALSE){
        X_2 = X_2 * 0 + doX2
      }
      X_3 = X_1 + 0.25 * X_2 + rnorm(n_obs)
    }
    
    
  }
  
  dat.s =  data.frame(x1 = X_1, x2 = X_2, x3 = X_3)
  dat.tf = tf$constant(as.matrix(dat.s), dtype = 'float32')
  A <- matrix(c(0, 1, 1, 0,0,1,0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
  
  if (is.null(dat_train)){
    scaled = scale_df(dat.tf)
  } else{
    scaled = scale_validation(dat_train, dat.tf)
  }
  return(list(df_orig=dat.tf, df_scaled = scaled, coef=coeffs, A=A, name='vaca2_triangle'))
} 

#Data from CAREFL Fig 5
if (USE_EXTERNAL_DATA){
  stop("Not implemted (no external data)")
} else{
  train = dgp(nTrain, seed=42)
}

pairs(train$df_orig$numpy())
par(mfrow=c(1,3))
hist(train$df_orig$numpy()[,1],100)
hist(train$df_orig$numpy()[,3],100)
hist(train$df_orig$numpy()[,2],100)
par(mfrow=c(1,1))
pairs(train$df_scaled$numpy())
train$A
library(igraph)
graph <- graph_from_adjacency_matrix(train$A, mode = "directed", diag = FALSE)
plot(graph, vertex.color = "lightblue", vertex.size = 30, edge.arrow.size = 0.5)

train_data = split_data(train$A, train$df_scaled)
tryCatch({
  # Try to call make_thetaNN
  thetaNN_l = make_thetaNN(train$A, train_data$parents)
}, error = function(e) {
  # If an error occurs, print the error message and try again
  cat("Error on first try:", e$message, "\n")
  cat("Trying again TF WTF...\n")
  thetaNN_l = make_thetaNN(train$A, train_data$parents)
})

if(USE_EXTERNAL_DATA){
  val = train
} else{
  val = dgp(5000, dat_train = train$df_orig)
}

val_data = split_data(val$A, val$df_scaled)

###### Training Step #####
optimizer = tf$keras$optimizers$Adam(learning_rate=0.001)
optimizer = tf$keras$optimizers$legacy$Adam(learning_rate=0.001)
l = do_training(train$name, thetaNN_l = thetaNN_l, train_data = train_data, val_data = val_data,
                SUFFIX, epochs = EPOCHS,  optimizer=optimizer)

### Save Script
dirname = paste0(DROPBOX, "exp/", train$name, "/", SUFFIX, "/")
file.copy(this_file, dirname)

loss = l[[1]]
loss_val = l[[2]]

plot(loss, type='l')
points(loss_val, col='green')

##### Plotting loss curve #####
# Create the ggplot
df <- data.frame(
  Epochs = seq_along(loss),
  loss = loss,
  loss_val = loss_val
)

data_long <- pivot_longer(df, cols = c(loss, loss_val), names_to = "Type", values_to = "Loss")

# Create the ggplot
g = ggplot(data_long, aes(x = Epochs, y = Loss, color = Type)) +
  geom_line() +
  ylim(-4.5, 0) +
  labs(
    x = "Epochs",
    y = "Loss"
  ) +
  theme_pubr()  
g
ggsave(make_fn("loss.pdf"))
if (FALSE){
  file.copy(make_fn("loss.pdf"), '~/Dropbox/Apps/Overleaf/tramdag/figures/', overwrite = TRUE)
}

#Loading data from epoch e
load_weights(epoch = EPOCHS, l)


#################################################
### Old Stuff Commented out #####################

if (FALSE){
  


################### Comparing the obs dist #######






plot_obs_fit(train_data$parents, train_data$target, thetaNN_l, name='Training')
plot_obs_fit(val_data$parents, val_data$target, thetaNN_l, name='Validation')

############################### Do X via Flow ########################
#Samples from Z give X=doX

############################
# Do Interventions on x #####
summary(train$df_orig[,1]$numpy())

if (DoX == 1){
  dox_origs = seq(-3, 4, by = 0.5)
} else if (DoX == 2){
  dox_origs = seq(-5, 5, by = 1)
}

num_samples = 25142L
inter_mean_dgp_x2 = inter_mean_dgp_x3 = inter_mean_ours_x2 = inter_mean_ours_x3 = NA*dox_origs
inter_dgp_x2 = inter_dgp_x3 = inter_ours_x2 = inter_ours_x3 = matrix(NA, nrow=length(dox_origs), ncol=num_samples)
for (i in 1:length(dox_origs)){
  dox_orig = dox_origs[i]
  #On X1
  if(DoX == 1){
    dox=scale_value(train$df_orig, col=1L, dox_orig) #On X1
    dat_do_x_s = do(thetaNN_l, train$A, doX = c(dox,NA,NA), num_samples = num_samples)
  } else{
    #On X2
    dox=scale_value(train$df_orig, col=2L, dox_orig) #On X2
    dat_do_x_s = do(thetaNN_l, train$A, doX = c(NA,dox,NA), num_samples = num_samples)
  }
  
  #TODO evt anpassen auf dox2
  df = unscale(train$df_orig, dat_do_x_s)
  inter_ours_x2[i,] = df$numpy()[,2]
  inter_ours_x3[i,] = df$numpy()[,3]
  inter_mean_ours_x2[i] = mean(df[,2]$numpy())
  inter_mean_ours_x3[i] = mean(df[,3]$numpy())
  #res_med_x4[i] = median(df[,4]$numpy())
  
  if(DoX == 1){
    d = dgp(num_samples,doX1=dox_orig)
  } else {
    d = dgp(num_samples,doX2=dox_orig)
  }
  inter_mean_dgp_x2[i] = mean(d$df_orig[,2]$numpy())
  inter_mean_dgp_x3[i] = mean(d$df_orig[,3]$numpy())
  inter_dgp_x2[i,] = d$df_orig[,2]$numpy()
  inter_dgp_x3[i,] = d$df_orig[,3]$numpy()
}

#Preparing a df for ggplot for selected do-values
df_do = data.frame(dox=numeric(0),x2=numeric(0),x3=numeric(0), type=character(0))
for (step in c(1,3,5,6,10)){
    df_do = rbind(df_do, data.frame(
      dox = dox_origs[step],
      x2 = inter_dgp_x2[step,],
      x3 = inter_dgp_x3[step,],
      type = 'simu'
    ))
    df_do = rbind(df_do, data.frame(
      dox = dox_origs[step],
      x2 = inter_ours_x2[step,],
      x3 = inter_ours_x3[step,],
      type = 'ours'
    )
  )
}

#### Adding the data from VACA2 ####
if (VACA == 2){
  X_inter <- read_csv("data/VACA2_triangle_nl/vaca2_triangle_nl_Xinter_x1-3.0.csv", col_names = FALSE)
  df_do = rbind(df_do, data.frame(
    dox = -3,
    x2 = X_inter$X2,
    x3 = X_inter$X3,
    type = 'CNF' 
  ))
  
  X_inter <- read_csv("data/VACA2_triangle_nl/vaca2_triangle_nl_Xinter_x1-2.csv", col_names = FALSE)
  df_do = rbind(df_do, data.frame(
    dox = -2,
    x2 = X_inter$X2,
    x3 = X_inter$X3,
    type = 'CNF' 
  ))
  
  X_inter <- read_csv("data/VACA2_triangle_nl/vaca2_triangle_nl_Xinter_x1-1.csv", col_names = FALSE)
  df_do = rbind(df_do, data.frame(
    dox = -1,
    x2 = X_inter$X2,
    x3 = X_inter$X3,
    type = 'CNF' 
  ))
  
  X_inter <- read_csv("data/VACA2_triangle_nl/vaca2_triangle_nl_Xinter_x1-0.5.csv", col_names = FALSE)
  df_do = rbind(df_do, data.frame(
    dox = -0.5,
    x2 = X_inter$X2,
    x3 = X_inter$X3,
    type = 'CNF' 
  ))
  
  X_inter <- read_csv("data/VACA2_triangle_nl/vaca2_triangle_nl_Xinter_x1+1.5.csv", col_names = FALSE)
  df_do = rbind(df_do, data.frame(
    dox = 1.5,
    x2 = X_inter$X2,
    x3 = X_inter$X3,
    type = 'CNF' 
  ))
} else if (VACA == 1){
  X_inter <- read_csv("data/VACA1_triangle_lin/25K/vaca1_triangle_lin_Xinter_x2=-5.csv", col_names = FALSE)
  df_do = rbind(df_do, data.frame(
    dox = -5,
    x2 = X_inter$X2,
    x3 = X_inter$X3,
    type = 'CNF' 
  ))
  
  X_inter <- read_csv("data/VACA1_triangle_lin/25K/vaca1_triangle_lin_Xinter_x2=-3.csv", col_names = FALSE)
  df_do = rbind(df_do, data.frame(
    dox = -3,
    x2 = X_inter$X2,
    x3 = X_inter$X3,
    type = 'CNF' 
  ))
  
  X_inter <- read_csv("data/VACA1_triangle_lin/25K/vaca1_triangle_lin_Xinter_x2=-1.csv", col_names = FALSE)
  df_do = rbind(df_do, data.frame(
    dox = -1,
    x2 = X_inter$X2,
    x3 = X_inter$X3,
    type = 'CNF' 
  ))
  
  X_inter <- read_csv("data/VACA1_triangle_lin/25K/vaca1_triangle_lin_Xinter_x2=0.csv", col_names = FALSE)
  df_do = rbind(df_do, data.frame(
    dox = 0,
    x2 = X_inter$X2,
    x3 = X_inter$X3,
    type = 'CNF' 
  ))
  
  X_inter <- read_csv("data/VACA1_triangle_lin/25K/vaca1_triangle_lin_Xinter_x2=4.csv", col_names = FALSE)
  df_do = rbind(df_do, data.frame(
    dox = 4,
    x2 = X_inter$X2,
    x3 = X_inter$X3,
    type = 'CNF' 
  ))
}


#Plotting the distribution of X2 variable when do on X1 
if (DoX == 1){#Makes no sense if on X2
  df_do$facet_label <- paste("dox1 =", df_do$dox)
  ggplot(df_do) + 
    geom_density(aes(x=x2, col=type, linetype=type)) + 
    ylab("p(x2|do(x1)") +
    facet_grid(~dox)
  ggsave(make_fn("dox1_dist_x2.pdf"))
}

ylab = "p(x3|do(x1)"
if (DoX == 2){
  ylab = "p(x3|do(x2)"
} 
filter(df_do, dox != '-5') %>% filter(dox != '4') %>% 
ggplot() + 
  geom_density(aes(x=x3, col=type, linetype=type), adjust = 1.5) + 
  xlim(-7,5) +
  ylab(ylab) +
  facet_grid(~dox)
if(DoX == 1){
  ggsave(make_fn("dox1_dist_x3.pdf"))
} else {
  ggsave(make_fn("dox2_dist_x3.pdf"), width = 15, height = 6)
  if (FALSE){
    file.copy(make_fn("dox2_dist_x3.pdf"), '~/Dropbox/Apps/Overleaf/tramdag/figures/', overwrite = TRUE)
  }
}

### Plotting the mean effects ####
library(ggpubr)
x1dat = data.frame(x=train$df_orig$numpy()[,1])
x2dat = data.frame(x=train$df_orig$numpy()[,1])

# X2
if (DoX == 1){
  df_do_mean = data.frame(
    x = dox_origs,
    Ours = inter_mean_ours_x2,
    #theoretical = dox_origs,
    Simulation_DGP = inter_mean_dgp_x2
  ) 
  d = df_do_mean %>% 
    pivot_longer(cols = 2:3)
  
  ggplot() + 
    geom_point(data = subset(d, name == "Ours"), aes(x=x, y=value, col=name)) +
    geom_line(data = subset(d, name == "Simulation_DGP"), aes(x=x, y=value, col=name))+
    #geom_abline(intercept = 0, slope = 1, col='skyblue') +
    xlab("do(X1)") +
    ylab("E(X2|do(X1)") +
    geom_rug(data=x1dat, aes(x=x), inherit.aes = FALSE, alpha=0.5) +
    theme_pubr() +  # Positioning the legend to the lower right corner
    labs(color = "")
  
  ggsave(make_fn("dox1_mean_x2.pdf"))
}


### X3
df_do_mean = data.frame(
  x = dox_origs,
  Ours = inter_mean_ours_x3,
  #theoretical = dox_origs,
  Simulation_DGP = inter_mean_dgp_x3
) 
d = df_do_mean %>% 
  pivot_longer(cols = 2:3)

if (DoX == 1){
  xlab = xlab("do(X1)")
  ylab = ylab("E(X3|do(X1)")
} else {
  xlab = xlab("do(X2)")
  ylab = ylab("E(X3|do(X2)")
}
ggplot() + 
  geom_point(data = subset(d, name == "Ours"), aes(x=x, y=value, col=name)) +
  geom_line(data = subset(d, name == "Simulation_DGP"), aes(x=x, y=value, col=name, type=name))+
  #geom_abline(intercept = 0, slope = 1, col='skyblue') +
  xlab + 
  ylab + 
  geom_rug(data=x1dat, aes(x=x), inherit.aes = FALSE, alpha=0.5)+
  theme_pubr() +  
  labs(color = "")

if (DoX == 1){
  ggsave(make_fn("dox1_mean_x3.pdf"))
} else {
  ggsave(make_fn("dox2_mean_x3.pdf"))
}


########
#### Counterfact ####
###############################################
# Counterfactual  
###############################################

if (VACA == 1){
  if(FALSE){
    X1 = -2 #A Typical value of X1 (linker Höcker)
    x1 = scale_value(dat_train_orig = train$df_orig, col = 1, value = X1)$numpy()
    x2 = get_x(thetaNN_l[[2]], x1, 0.)
    x3 = get_x(thetaNN_l[[3]], c(x1,x2), 0.)
    df = data.frame(x1=x1,x2=as.numeric(x2), x3=as.numeric(x3))
    unscaled = unscale(train$df_orig, tf$constant(as.matrix(df), dtype=tf$float32))$numpy()
    #-2 1.954844 -1.514063
    X2 = unscaled[2]
    X3 = unscaled[3]
    #Next Neighbor execute code below
  }
  
  #TODO DELETE
  X1 = 2.260561 
  X2 = -2.628146 
  X3 = 4.047885
  X_obs = c(X1, X2, X3)
  xobs = scale_validation(train$df_orig, X_obs)$numpy()
  
  X1 = -1.951140  
  X2 = 1.950804 
  X3 = -1.540089
  X_obs = c(X1, X2, X3)
  xobs = scale_validation(train$df_orig, X_obs)$numpy()
  
  cf_do_x2_dgp = function(alpha, X1, X2, X3){
    #Abduction these are the Us that correspond the observed values
    #lambda x1: x1,
    #lambda x1, x2: (x1 + x2),
    #lambda x1, x2, x3: (x3 - 0.25 * x2 - x1),
    
    U1 = X1
    U2 = X1 + X2
    U3 = X3 - 0.25 * X2 - X1
    
    #Counterfactual X2 --> alpha
    X_1 = X1
    X_2 = alpha 
    X_3 = X_1 + 0.25*X_2 + U3
    return(data.frame(X1=X_1,X2=X_2,X3=X_3))
  }
  
  #Constistency
  cf_dgp_cons = cf_do_x2_dgp(alpha=X2, X1, X2, X3)
  abs(cf_dgp_cons$X1-X1) #~1e-16 Consistency
  abs(cf_dgp_cons$X2-X2) #~1e-16 Consistency
  abs(cf_dgp_cons$X3-X3) #~1e-16 Consistency
  
  #
  num_samples = 1000
  df = data.frame()
  for (a_org in c(seq(-6,6,0.01),X2)){
    dpg = cf_do_x2_dgp(a_org,X1,X2,X3)
    df = bind_rows(df, data.frame(x1=dpg[1], X2=a_org, X3=dpg[3], type='DGP'))
  }
  
  #From the model
  #Constistency
  computeCF(thetaNN_l, A=train$A, xobs = xobs, cfdoX = c(NA, NA,NA)) - xobs
  computeCF(thetaNN_l, A=train$A, xobs = xobs, cfdoX = c(xobs[1], NA,NA)) - xobs
  computeCF(thetaNN_l, A=train$A, xobs = xobs, cfdoX = c(NA, xobs[2],NA)) - xobs
  computeCF(thetaNN_l, A=train$A, xobs = xobs, cfdoX = c(NA, NA,xobs[3])) - xobs
  
  for (a_org in c(seq(-6,6,0.5),X2)){
    a = scale_value(train$df_orig, 2L, a_org)$numpy()
    printf("a_org %f a %f \n", a_org, a)
    #cf_our = cf_do_x1_ours(a_org)
    cf_our = computeCF(thetaNN_l = thetaNN_l, A = train$A, cfdoX = c(NA,a,NA), xobs = xobs)
    cf_our = unscale(train$df_orig, matrix(cf_our, nrow=1))$numpy()
    df = bind_rows(df, data.frame(x1=cf_our[1], X2=a_org, X3=cf_our[3], type='OURS'))
  }
  
  ggplot(df) +
    geom_point(data = subset(df, type == "OURS"), aes(x = X2, y = X3, color=type)) +
    geom_line(data = subset(df, type == "DGP"), aes(x = X2, y = X3, color=type)) + 
    xlab('would x2 have been alpha')  + 
    #geom_rug(data=x2dat, aes(x=x), inherit.aes = FALSE, alpha=0.5) +
    theme_pubr() +  # Positioning the legend to the lower right corner
    labs(color = "") 
  
  ggsave(make_fn("CFx2_x3.pdf"))

  #Looping over the training data to calculate the CF_RMSE as in VACA2:Tab2
  a_org = 2.
  a = scale_value(train$df_orig, 2L, a_org)$numpy()
  res_x3 = rep(NA, 100) #rep(NA, nTrain)
  for (i in 1:length(res_x3)){
    X = train$df_orig[i,]$numpy()
    x = train$df_scaled[i,]$numpy()
    cf_true = cf_do_x2_dgp(alpha=a_org, X[1], X[2], X[3])
    
    cf_model = computeCF(thetaNN_l = thetaNN_l, A = train$A, cfdoX = c(NA,a,NA), xobs = x)
    cf_model = unscale(train$df_orig, matrix(cf_model, nrow=1))$numpy()
    res_x3[i] = (cf_true - cf_model)[3]
  }
  hist(unlist(res_x3)^2,10)
  
}#VACA1

if (VACA == 2){
# CF of X4 given X1=alpha ###

#Creating a typical value 
if (FALSE){
  X1 = 1.
  x1 = scale_value(dat_train_orig = train$df_orig, col = 1, value = X1)$numpy()
  x2 = get_x(thetaNN_l[[2]], x1, 0.)
  x3 = get_x(thetaNN_l[[3]], c(x1,x2), 0.)
  df = data.frame(x1=0,x2=as.numeric(x2), x3=as.numeric(x3))
  unscaled = unscale(train$df_orig, tf$constant(as.matrix(df), dtype=tf$float32))$numpy()
  X2 = unscaled[2]
  X3 = unscaled[3]
  
  dat = val$df_orig$numpy()
  point = c(X1,X2,X3)
  distances <- apply(dat, 1, function(row) sqrt(sum((row - point)^2)))
  # Find the index of the closest row
  closest_row_index <- which.min(distances)
  # Retrieve the closest row
  closest_row <- dat[closest_row_index,]
  closest_row #1.081913  1.929156 18.40600
} else{
  X1 = 1.081913
  X2 = 1.929156
  X3 = 18.406000
  X_obs = c(X1, X2, X3)
}

###### CF Theoretical (assume we know the complete SCM) #####
cf_do_x1_dgp = function(alpha){
  #Abduction these are the Us that correspond the observed values
  U1 = X1 - 1
  U2 = X2 - 2*X1^2
  U3 = X3 - 20./(1 + exp(-X2^2 + X1)) 
  
  #X1 --> alpha
  X_1 = alpha
  X_2 = 2*X_1^2 + U2 
  X_3 = 20./(1 + exp(-X_2^2 + X_1)) + U3
  return(data.frame(X1=X_1,X2=X_2,X3=X_3))
}

#Constistency
cf_dgp_cons = cf_do_x1_dgp(X1)
abs(cf_dgp_cons$X2-X2) #~1e-16 Consistency
abs(cf_dgp_cons$X3-X3) #~1e-16 Consistency
alpha = seq(-3,3,0.1)

##### Our Approach ####
xobs = scale_validation(train$df_orig, X_obs)$numpy()
computeCF(thetaNN_l, A=train$A, xobs = xobs, cfdoX = c(NA, NA,NA)) - xobs
computeCF(thetaNN_l, A=train$A, xobs = xobs, cfdoX = c(xobs[1], NA,NA)) - xobs
computeCF(thetaNN_l, A=train$A, xobs = xobs, cfdoX = c(NA, xobs[2],NA)) - xobs
computeCF(thetaNN_l, A=train$A, xobs = xobs, cfdoX = c(NA, NA,xobs[3])) - xobs

## Creating Results for do(x1)
df = data.frame()
num_samples = 1000
for (a_org in c(seq(-3,3,0.05),X1)){
  dpg = cf_do_x1_dgp(a_org)
  df = bind_rows(df, data.frame(x1=a_org, X2=dpg[2], X3=dpg[3], type='DGP'))
}

for (a_org in c(seq(-3,3,0.2),X1)){
  a = scale_value(train$df_orig, 1L, a_org)$numpy()
  printf("a_org %f a %f \n", a_org, a)
  #cf_our = cf_do_x1_ours(a_org)
  cf_our = computeCF(thetaNN_l = thetaNN_l, A = train$A, cfdoX = c(a,NA,NA), xobs = xobs)
  cf_our = unscale(train$df_orig, matrix(cf_our, nrow=1))$numpy()
  df = bind_rows(df, data.frame(x1=a_org, X2=cf_our[2], X3=cf_our[3], type='OURS'))
}

ggplot(df) +
  geom_point(data = subset(df, type == "OURS"), aes(x = x1, y = X2, color=type)) +
  geom_line(data = subset(df, type == "DGP"), aes(x = x1, y = X2, color=type)) + 
  xlab('would x1 be alpha')  + 
  geom_rug(data=x1dat, aes(x=x), inherit.aes = FALSE, alpha=0.5) +
  theme_pubr() +  # Positioning the legend to the lower right corner
  labs(color = "") 

ggsave(make_fn("CFx1_x2.pdf"))
#

ggplot(df) +
  geom_point(data = subset(df, type == "OURS"), aes(x = x1, y = X3, color=type)) +
  geom_line(data = subset(df, type == "DGP"), aes(x = x1, y = X3, color=type)) + 
  xlab('would x1 be alpha')  + 
  geom_rug(data=x1dat, aes(x=x), inherit.aes = FALSE, alpha=0.5) +
  theme_pubr() +  # Positioning the legend to the lower right corner
  labs(color = "") 

ggsave(make_fn("CFx1_x3.pdf"))
}#VACA2


#####################
# Old code to check Do intervention
if (FALSE){
  dox1 = function(doX, thetaNN_l, num_samples){
    doX_tensor = doX * tf$ones(shape=c(num_samples,1L),dtype=tf$float32) 
    
    x2_samples = sample_from_target(thetaNN_l[[2]], doX_tensor)
    
    parents_x3 = tf$concat(list(doX_tensor, x2_samples), axis=1L)
    x3_samples = sample_from_target(thetaNN_l[[3]], parents_x3)
    
    return(matrix(c(doX_tensor$numpy(),x2_samples$numpy(), x3_samples$numpy()), ncol=3))
  }
  
  df = dox1(0.5, thetaNN_l, num_samples=1e4L)
  str(df)
  summary(df)
  
  df2t = do(thetaNN_l, train$A, doX=c(0.5, NA, NA), num_samples=1e4)
  df2 = as.matrix(df2t$numpy())
  qqplot(df2[,2], df[,2]);abline(0,1)
  qqplot(df2[,3], df[,3]);abline(0,1)
  qqplot(df2[,4], df[,4]);abline(0,1)
}


}






