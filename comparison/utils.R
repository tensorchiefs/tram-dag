##################################
##### Utils for tram_dag #########
library(mlt)
library(tram)
library(MASS)
library(tensorflow)
library(keras)
library(tidyverse)
#library(tfprobability)

source('comparison/bern_utils.R')
source('comparison/model_utils.R')

version_info = function(){
  print(reticulate::py_config())
  print('R version:')
  print('tensorflow version:')
  print(tf$version$VERSION)
  print('keras version:')
  print(reticulate::py_run_string("import keras; print(keras.__version__)"))
  print('tfprobability version:')
  print(tfprobability::tfp_version())
}

#' Training for loading a cached version
#'
#' @param name the name of the data file
#' @param thetaNN_l the (untrained) list of weights for the network
#' @param train_data the training data
#' @param val_data the validation data
#' @param SUFFIX a name for the current run, describing some details like SUFFIX = 'runLaplace_M10_C0.5_long3K_M30_nTrain500'
#' @param epochs Number of epochs to train 
#' @param optimizer The optimizer to use
#' @param dynamic_lr TRUE If the learning rate should be dynamically adapted (@seealso [update_learning_rate()]) 
#'
#' @return the of the loss list(loss, loss_val)
#'
#' @examples 
do_training = function(name, thetaNN_l, train_data, val_data, 
                       SUFFIX, epochs=200, 
                       optimizer= tf$keras$optimizers$Adam(learning_rate=0.001),
                       dynamic_lr = TRUE
){
  parents_l = train_data$parents
  target_l = train_data$target
  
  parents_l_val = val_data$parents
  target_l_val = val_data$target
  
  
  loss = loss_val = rep(NA, epochs)
  dirname = paste0(DROPBOX, "exp/", train$name, "/", SUFFIX, "/")
  create_directories_if_not_exist(dirname)
  loss_name = paste0(dirname, 'losses.rda')
  
  if(file.exists(loss_name) == FALSE){
    for (e in 1:epochs){
      l = train_step(thetaNN_l, parents_l, target_l, optimizer=optimizer)
      
      loss[e]  = l$NLL$numpy()
      
      NLL_val = 0  
      for(i in 1:ncol(val$A)) { # Assuming that thetaNN_l, parents_l and target_l have the same length
        NLL_val = NLL_val + calc_NLL(thetaNN_l[[i]], parents_l_val[[i]], target_l_val[[i]])$numpy()
      }
      loss_val[e] = NLL_val
      
      if (dynamic_lr){
        update_learning_rate(optimizer,NLL_val)
      }
      printf("e:%f  Train: %f, Val: %f \n",e, l$NLL$numpy(), NLL_val)
      if (e %% 10 == 0) {
        
        for (i in 1:ncol(val$A)){
          #printf('Layer %d checksum: %s \n',i, calculate_checksum(thetaNN_l[[i]]$get_weights()))
          
          #There might be a problem with h5
          #fn = paste0(dirname,train$name, "_nn", i, "_e", e, "_weights.h5")
          #thetaNN_l[[i]]$save_weights(path.expand(fn))
          
          fn = paste0(dirname,train$name, "_nn", i, "_e", e, "_weights.rds")
          saveRDS(thetaNN_l[[i]]$get_weights(), path.expand(fn))
          
          #fn = paste0(dirname,train$name, "_nn", i, "_e", e, "_model.h5")
          #save_model_hdf5(thetaNN_l[[i]], fn)
          
          fn = paste0(dirname,train$name, "_nn", i, "_e", e, "_checksum.txt")
          write(calculate_checksum(thetaNN_l[[i]]$get_weights()), fn)
        }
      }
    }
    # Saving weights after training
    save(loss, loss_val, file=loss_name)
  }  else {
    print('Loss File Exists')
  }
  load(loss_name)
  return(list(loss, loss_val))
}


#' Splits the data into parents and targets
#'
#' @param A the adjacency matrix
#' @param dat_scaled.tf 
#'
#' @return list of parents and targets
#'
#' @examples
split_data = function(A, dat_scaled.tf){
  parents_l = target_l = list()
  for (i in 1:ncol(A)){
    parents = which(A[,i] == 1)
    if (length(parents) == 0){ # No parents source node
      parents_tmp = 0*dat_scaled.tf[,i] + 1
      #thetaNN_tmp = make_model(len_theta, 1)
      target_tmp = dat_scaled.tf[,i, drop=FALSE]
    } else{ # Node has parents
      parents_tmp  = dat_scaled.tf[,parents,drop=FALSE]
      #thetaNN_tmp = make_model(len_theta, length(parents))
      target_tmp = dat_scaled.tf[,i, drop=FALSE]
    }
    parents_l = append(parents_l, parents_tmp)
    #thetaNN_l = append(thetaNN_l, thetaNN_tmp)
    target_l = append(target_l, target_tmp)
  }
  return(list(parents = parents_l, target = target_l))
}

###### Bulding needed networks 
make_thetaNN = function(A, parents_l){
  thetaNN_l = list()
  for (i in 1:length(parents_l)){
    parents = which(A[,i] == 1)
    if (length(parents) == 0){
      thetaNN_l[[i]] = make_model(len_theta, 1)
    } else{
      thetaNN_l[[i]] = make_model(len_theta, length(parents))
    }
  }
  return(thetaNN_l)
}

update_learning_rate <- function(optimizer, current_loss, factor=0.1, patience=50, min_lr=1e-7) {
  if (!exists("best_loss")) {
    assign("best_loss", current_loss, envir=globalenv())
    assign("wait", 0, envir=globalenv())
  }
  
  if (current_loss < best_loss) {
    assign("best_loss", current_loss, envir=globalenv())
    assign("wait", 0, envir=globalenv())
  } else {
    wait <<- wait + 1
    if (wait >= patience) {
      current_lr <- optimizer$learning_rate$numpy()
      new_lr <- max(current_lr * factor, min_lr)
      optimizer$learning_rate$assign(new_lr)
      cat("\033[31mReduced learning rate to", new_lr, "\033[0m\n")
      assign("wait", 0, envir=globalenv())
    }
  }
}

train_step = function(thetaNN_l, parents_l, target_l, optimizer){
  n = length(thetaNN_l)
  
  with(tf$GradientTape() %as% tape, {
    NLL = 0  # Initialize NLL
    for(i in 1:n) { # Assuming that thetaNN_l, parents_l and target_l have the same length
      NLL = NLL + calc_NLL(thetaNN_l[[i]], parents_l[[i]], target_l[[i]], target_index=i)
    }
  })
  
  tvars = list()
  for(i in 1:n) {
    tvars[[i]] = thetaNN_l[[i]]$trainable_variables
  }
  
  #Calculation of the gradients
  grads = tape$gradient(NLL, tvars)
  for (i in 1:n){
    optimizer$apply_gradients(
      purrr::transpose(list(grads[[i]], tvars[[i]]))
    )  
  }
  
  return(list(NLL=NLL))
}

create_directories_if_not_exist <- function(dir_path) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
    printf("Creating : %s \n", dir_path)
  }
}

calc_NLL = function(nn_theta_tile, parents, target, target_index=NULL){
  if (!is.null(target_index)) {
   if (model_tagettype == 'cont') {
     ### Modelling the intercept
     cis = which(model_A[,target_index] == 'ci')
     if(length(cis) == 0) { #No complex intercept
       ones = tf$ones(shape=c(parents$shape[1],1L),dtype=tf$float32)  
       theta_tilde = nn_theta_tile(ones)
       theta_tilde = tf$cast(theta_tilde, dtype=tf$float32)
       theta = to_theta(theta_tilde)
     }
     
     if (model_A[,] == 'ls') {
       
     } else{
       
     }
   } else{
     print("Not implemented yet~~~~~~~~~~")
   }
  } else{ #Old behavior
    theta_tilde = nn_theta_tile(parents)
    theta_tilde = tf$cast(theta_tilde, dtype=tf$float32)
    theta = to_theta(theta_tilde)
    
    #latentold = eval_h(theta, y_i = target, beta_dist_h = bp$beta_dist_h)
    latent = eval_h_extra(theta, y_i = target, beta_dist_h = bp$beta_dist_h, beta_dist_h_dash = bp$beta_dist_h_dash)
    
    #h_dashOld = eval_h_dash(theta, target, beta_dist_h_dash = bp$beta_dist_h_dash)
    h_dash = eval_h_dash_extra(theta, target, beta_dist_h_dash = bp$beta_dist_h_dash)
  }
  
  
  pz = latent_dist
  return(
    -tf$math$reduce_mean(
      pz$log_prob(latent) +
        tf$math$log(h_dash))
  )
  # return(
  #   -tfp$stats$percentile(
  #     pz$log_prob(latent) + 
  #       tf$math$log(h_dash), 50.)
  # )
}


predict_p_target = function(thetaNN, parents, target_grid){  theta_tilde = thetaNN(parents)
  theta_tilde = tf$cast(theta_tilde, dtype=tf$float32)
  theta = to_theta(theta_tilde)
  latent = eval_h_extra(theta, y_i = target_grid, beta_dist_h = bp$beta_dist_h, beta_dist_h_dash = bp$beta_dist_h_dash)
  h_dash = eval_h_dash_extra(theta, target_grid, beta_dist_h_dash = bp$beta_dist_h_dash)
  pz = latent_dist
  p_target = pz$prob(latent) * h_dash
  return(p_target)
}

predict_h = function(thetaNN, parents, target_grid){
  theta_tilde = thetaNN(parents)
  theta_tilde = tf$cast(theta_tilde, dtype=tf$float32)
  theta = to_theta(theta_tilde)
  latent = eval_h_extra(theta, y_i = target_grid, beta_dist_h = bp$beta_dist_h, beta_dist_h_dash= bp$beta_dist_h_dash)
  return(latent)
}

sample_within_bounds <- function(h_0, h_1) {
  samples <- map2_dbl(h_0, h_1, function(lower_bound, upper_bound) {
    while(TRUE) {
      sample <- as.numeric(latent_dist$sample())
      if (lower_bound < sample && sample < upper_bound) {
        return(sample)
      }
    }
  })
  return(samples)
}

sample_from_target = function(thetaNN, parents){
  theta_tilde = thetaNN(parents)
  theta_tilde = tf$cast(theta_tilde, dtype=tf$float32)
  theta = to_theta(theta_tilde)
  h_0 =  tf$expand_dims(eval_h(theta, L_START, beta_dist_h = bp$beta_dist_h), axis=1L)
  h_1 = tf$expand_dims(eval_h(theta, R_START, beta_dist_h = bp$beta_dist_h), axis=1L)
  if (DEBUG_NO_EXTRA){
    s = sample_within_bounds(h_0$numpy(), h_1$numpy())
    latent_sample = tf$constant(s)
    if(FALSE){
      h_0 =  tf$expand_dims(eval_h(theta, 0.01, beta_dist_h = bp$beta_dist_h), axis=1L)
      h_1 = tf$expand_dims(eval_h(theta, 0.99, beta_dist_h = bp$beta_dist_h), axis=1L)
      h_0_2 = tf$squeeze(tf$concat(c(h_0, h_0), axis=0L))
      h_1_2 = tf$squeeze(tf$concat(c(h_1, h_1), axis=0L))
      len = as.numeric(theta_tilde$shape[1])
      l = latent_dist$sample(h_0_2$shape[1])
      # Get the boolean mask where condition is true
      mask = tf$math$logical_and(l >= h_0_2, l <= h_1_2)
      
      
      
      # Use boolean mask to get the values
      latent_sample = tf$boolean_mask(l, mask)[1:len]
    }
    
    
  } else { #The normal case allowing extrapolations
    latent_sample = latent_dist$sample(theta_tilde$shape[1])
  }
  
  #  object_fkt = function(t_i){
  #     return(tf$reshape((eval_h(theta, y_i = t_i, beta_dist_h = bp$beta_dist_h) - latent_sample), c(theta_tilde$shape[1],1L)))
  # }
  # shape = tf$shape(parents)[1]
  # target_sample1 = tfp$math$find_root_chandrupatla(object_fkt, low = 0, high = 1)$estimated_root
  # target_sample1
  
  object_fkt = function(t_i){
    return(tf$reshape((eval_h_extra(theta, y_i = t_i, beta_dist_h = bp$beta_dist_h,beta_dist_h_dash = bp$beta_dist_h_dash) - latent_sample), c(theta_tilde$shape[1],1L)))
  }
  shape = tf$shape(parents)[1]
  #target_sample = tfp$math$find_root_chandrupatla(object_fkt, low = -1E5*tf$ones(c(shape,1L)), high = 1E5*tf$ones(c(shape,1L)))$estimated_root
  target_sample = tfp$math$find_root_chandrupatla(object_fkt, low = h_0, high = h_1)$estimated_root
  
  # Manuly calculating the inverse for the extrapolated samples
  ## smaller than h_0
  l = tf$expand_dims(latent_sample, 1L)
  mask <- tf$math$less_equal(l, h_0)
  printf('~~~ sample_from_target  Fraction of extrapolated samples < 0 : %f \n', tf$reduce_mean(tf$cast(mask, tf$float32)))
  #tf$where(mask, beta_dist_h$prob(y_i)* theta_im, h)
  slope0 <- tf$expand_dims(eval_h_dash(theta, 0., bp$beta_dist_h_dash), axis=1L)
  target_sample = tf$where(mask, (l-h_0)/slope0, target_sample)
  
  ## larger than h_1
  mask <- tf$math$greater_equal(l, h_1)
  #tf$where(mask, beta_dist_h$prob(y_i)* theta_im, h)
  slope1<- tf$expand_dims(eval_h_dash(theta, 1., bp$beta_dist_h_dash), axis=1L)
  target_sample = tf$where(mask, (l-h_1)/slope1 + 1.0, target_sample)
  printf('sample_from_target Fraction of extrapolated samples > 1 : %f \n', tf$reduce_mean(tf$cast(mask, tf$float32)))
  return(target_sample)
}


unscale = function(dat_train_orig, dat_scaled){
  # Get original min and max
  orig_min = tf$reduce_min(dat_train_orig, axis=0L)
  orig_max = tf$reduce_max(dat_train_orig, axis=0L)
  dat_scaledtf = tf$constant(as.matrix(dat_scaled), dtype = 'float32')
  # Reverse the scaling
  return(dat_scaledtf * (orig_max - orig_min) + orig_min)
}

scale_df = function(dat_tf){
  dat_min = tf$reduce_min(dat_tf, axis=0L)
  dat_max = tf$reduce_max(dat_tf, axis=0L)
  dat_scaled = (dat_tf - dat_min) / (dat_max - dat_min)
  return(dat_scaled)
}

scale_validation = function(dat_training, dat_val){
  dat_min = tf$reduce_min(dat_training, axis=0L)
  dat_max = tf$reduce_max(dat_training, axis=0L)
  dat_scaled = (dat_val - dat_min) / (dat_max - dat_min)
  return(dat_scaled)
}

scale_value = function(dat_train_orig, col, value){
  # Get original min and max
  orig_min = tf$reduce_min(dat_train_orig[,col], axis=0L)
  orig_max = tf$reduce_max(dat_train_orig[,col], axis=0L)
  # Reverse the scaling
  return((value - orig_min) / (orig_max - orig_min))
}


make_model_old = function(len_theta, parent_dim){ 
  model <- keras_model_sequential() 
  model %>% 
    layer_dense(units=(10), input_shape = c(parent_dim), activation = 'tanh') %>% 
    layer_dense(units=(100), activation = 'tanh') %>% 
    layer_dense(units=len_theta) %>% 
    layer_activation('linear') 
  return (model)
}

make_model <- function(len_theta, parent_dim) { 
  model <- keras_model_sequential() 
  model$add(layer_dense(units = 10, input_shape = c(parent_dim), activation = 'tanh'))
  model$add(layer_dense(units = 100, activation = 'tanh'))
  model$add(layer_dense(units = len_theta, activation = 'linear'))
  return(model)
}



# Function to calculate the SHA256 checksum
library(digest)
calculate_checksum <- function(weights) {
  # Start an empty vector to hold all byte-converted weights
  weights_bytes <- c()
  
  for (i in 1:length(weights)) {
    # Transform the weight to bytes
    weight_bytes <- serialize(weights[[i]], NULL)
    weights_bytes <- c(weights_bytes, weight_bytes)
  }
  
  # Calculate the digest on the concatenated byte strings
  return(digest(weights_bytes, algo="sha256", serialize=FALSE))
}

plot_obs_fit = function(parents_l, target_l, thetaNN_l,name){
"' Please note that this samples are generated using the **given observed** parents
"
  for (i in 1:length(parents_l)){
    parents = parents_l[[i]]
    targets = target_l[[i]]
    thetaNN = thetaNN_l[[i]]
    hist(targets$numpy(), freq = FALSE,100, main=paste0(name, ' x_',i, ' green for model'))
    x_samples = sample_from_target(thetaNN_l[[i]], parents)
    lines(density(x_samples$numpy()), col='green')
    #hist(x_samples$numpy(),100,xlim=c(0,1))
  }
}


###### Helper Functions (simple for non-time critical stuff)
#@static_method
get_z = function(net, parents, x){
  parents = tf$constant(matrix(parents, nrow=1), dtype = tf$float32)
  x = tf$constant(x)
  theta_tilde = net(parents)
  theta = to_theta(theta_tilde)
  res =  eval_h_extra(theta, x, beta_dist_h = bp$beta_dist_h, beta_dist_h_dash = bp$beta_dist_h_dash)
  return(as.numeric(res$numpy()))
}

#' get_x return the x for a given z. 
#' simple function for non-time critical stuff
#'
#' More detailed description of the function.
#' @param net the network (the last layer net)
#' @return The value of x
#' 
#@static_method
get_x = function(net, parents, z){
  parents = tf$constant(matrix(parents, nrow=1), dtype = tf$float32)
  z = tf$constant(z)
  theta_tilde = net(parents)
  theta = to_theta(theta_tilde)
  ## Prediction
  latent_sample = z
  object_fkt = function(t_i){
    return(tf$reshape((eval_h_extra(theta, y_i = t_i, 
                                    beta_dist_h = bp$beta_dist_h,
                                    beta_dist_h_dash = bp$beta_dist_h_dash) - latent_sample), 
                      c(theta_tilde$shape[1],1L)))
  }
  res = tfp$math$find_root_chandrupatla(object_fkt, low = 0., high = 1.)$estimated_root
  return(as.numeric(res$numpy()))
}

########### Interventions ######
#### Helper ####
is_upper_triangular <- function(mat) {
  # Ensure it's a square matrix
  if (nrow(mat) != ncol(mat)) {
    return(FALSE)
  }
  
  # Check if elements below the diagonal are zero
  for (i in 1:nrow(mat)) {
    for (j in 1:ncol(mat)) {
      if (j < i && mat[i, j] != 0) {
        return(FALSE)
      }
      if (j == i && mat[i, j] != 0) {
        return(FALSE)
      }
    }
  }
  
  return(TRUE)
}

##### Observational Distribution ########
#' Draws samples from a tramDAG (in scaled space!)
#'
#' @param thetaNN_l The list of weights of the networks
#' @param A The Adjacency Matrix (first column is parents for x1 and hence empty)
#' @param doX The variables on which the do operation should be done and the 
#' strength of the intervention (in the scaled space). NA indicates no interaction
#' @num_samples  The number of samples to be drawn
#' @return Returns samples for the defined intervention (num_samples x N)
#' 
#' @examples
#' sampleObs(thetaNN_l, A, 1000) draws 1000 samples of the observational study from the model
sampleObs = function(thetaNN_l, A, num_samples=1042){
  doX = rep(NA, rep(nrow(A)))
  return(do(thetaNN_l=thetaNN_l, A=A, doX=doX,num_samples=num_samples))
}


##### Do Interventions ########
#' Draws samples from a tramDAG using the do-operation (in scaled space!)
#'
#' @param thetaNN_l The list of weights of the networks
#' @param A The Adjacency Matrix (first column is parents for x1 and hence empty)
#' @param doX The variables on which the do operation should be done and the 
#' strength of the intervention (in the scaled space). NA indicates no interaction
#' @num_samples  The number of samples to be drawn
#' @return Returns samples for the defined intervention (num_samples x N)
#' 
#' @examples
#' do(thetaNN_l, doX = c(0.5, NA, NA), 1000) draws 1000 samples from do(x_1=0.5)
#' do(thetaNN_l, doX = c(NA, NA, NA), 1000) draws 1000 samples from observational distribution

do = function(thetaNN_l, A, doX = c(0.5, NA, NA, NA), num_samples=1042){
  num_samples = as.integer(num_samples)
  N = length(doX)
  
  #### Checking the input #####
  stopifnot(is_upper_triangular(A)) #A needs to be upper triangular
  stopifnot(length(thetaNN_l) == N) #Same number of variables
  stopifnot(nrow(A) == N)           #Same number of variables
  stopifnot(sum(is.na(doX)) >= N-1) #Currently only one Variable with do(might also work with more but not tested)
  
  # Looping over the variables assuming causal ordering
  #Sampling (or replacing with do) of the current variable x
  xl = list() 
  for (i in 1:N){
    x = NA
    parents = which(A[,i] == 1)
    if (length(parents) == 0) { #Root node?
      ones = tf$ones(shape=c(num_samples,1L),dtype=tf$float32)
      if(is.na(doX[i])){ #No do ==> replace with samples (conditioned on 1)
        x = sample_from_target(thetaNN_l[[i]], ones)
      } else{
        x = doX[i] * ones #replace with do
      }
    } else { #No root node ==> the parents are present 
      if(is.na(doX[i])){ #No do ==> replace with samples (conditioned on 1)
        x = sample_from_target(thetaNN_l[[i]], tf$concat(xl[parents], axis = 1L))
      } else{ #Replace with do
        ones = tf$ones(shape=c(num_samples,1L),dtype=tf$float32) 
        x = doX[i] * ones #replace with do
      }
    }
    xl = c(xl, x)
  }
  
  return(tf$concat(xl, axis = 1L))
}

##### CF Interventions ########
#' Calculates the counter-factual values from a tramDAG using(in scaled space!)
#'
#' @param thetaNN_l The list of weights of the networks
#' @param A The Adjacency Matrix (first column is parents for x1 and hence empty)
#' @param Xobs the observed factual value (in the scaled space).
#' @param doX The variables on which the counter factual do operation should be done and the 
#' strength of the intervention (in the scaled space). NA indicates no interaction
#' @num_samples  The number of samples to be drawn
#' @return Returns samples for the defined intervention (num_samples x N)

computeCF = function(thetaNN_l, A, xobs, cfdoX = c(0.5, NA, NA, NA)){
  N = length(cfdoX)
  
  #### Checking the input
  stopifnot(is_upper_triangular(A)) #A needs to be upper triangular
  stopifnot(length(thetaNN_l) == N) #Same number of variables
  stopifnot(nrow(A) == N)           #Same number of variables
  stopifnot(length(xobs) == N)           #Same number of variables
  stopifnot(sum(is.na(cfdoX)) >= N-1) #Currently only one Variable with do (might also work with more but not tested)
  
  ### Abduction Getting the latent variable
  zobs_l = list() 
  for (i in 1:N){
    parents = which(A[,i] == 1)
    if (length(parents) == 0) { #Root node?
      z = get_z(thetaNN_l[[i]], parents = 1L, x=xobs[i])
    } else { #No root node ==> the parents are present 
      z = get_z(thetaNN_l[[i]], parents = tf$concat(xobs[parents],axis=0L), x=xobs[i])
    }
    zobs_l = c(zobs_l, z)
  }
  
  ## Action: Replace the observation with the counterfactuals ones where applicable
  x_cf = xobs 
  cfdo_idx = !is.na(cfdoX)
  x_cf[cfdo_idx] = cfdoX[cfdo_idx]
  
  ## Prediction: Predict the remaining ones.
  for (i in which(!cfdo_idx)){
    parents = which(A[,i] == 1)
    if (length(parents) == 0) { #Root node?
      x = get_x(thetaNN_l[[i]], parents = 1L, z=zobs_l[i]) #Will be the same as x_obs
    } else{
      x = get_x(net=thetaNN_l[[i]], parents = tf$concat(x_cf[parents],axis=0L), zobs_l[i])
    }
    x_cf[i] = x
  }
  
  return(x_cf)
}




########### Loading #########
load_weights = function(epoch, l){
###
  #' epoch number of epoch to load the data
  #' l loss list as returned by do_training 
###
  printf("Loading previously stored weight of epoch: %d\n", epoch)
  e = epoch
  loss = l[[1]]
  loss_val = l[[2]]
  
  #e = EPOCHS
  for (i in 1:ncol(val$A)){
    fn = paste0(dirname,train$name, "_nn", i, "_e", e, "_weights.rds")
    thetaNN_l[[i]]$set_weights(readRDS(fn))
    
    checksum = calculate_checksum(thetaNN_l[[i]]$get_weights())
    
    fn_checksum = paste0(dirname,train$name, "_nn", i, "_e", e, "_checksum.txt")
    file_content <- readLines(fn_checksum)
    if (checksum == file_content){
      printf('Layer %d checksum consistent with stored \n',i)
    } else {
      printf('Layer %d checksum calculated %s\n',i, checksum)
      printf('Layer %d checksum stored %s\n',i, file_content)
      stop('checksum missmatch')
    }
  }
  
  NLL_val = NLL_train = NLL_val2 = 0  
  for(i in 1:ncol(val$A)) { # Assuming that thetaNN_l, parents_l and target_l have the same length
    #nn_theta_tile, parents, target, target_index=NULL
    NLL_train = NLL_train + calc_NLL(
        nn_theta_tile = thetaNN_l[[i]], 
        parents = train_data$parents[[i]], 
        target = train_data$target[[i]])$numpy()
    NLL_val = NLL_val + calc_NLL(thetaNN_l[[i]], val_data$parents[[i]], val_data$target[[i]])$numpy()
    #NLL_val2 = NLL_val2 + calc_NLL(thetaNN_l[[i]], parents_l_val2[[i]], target_l_val2[[i]])$numpy()
  }
  
  printf('loss loaded training:     %f \n', loss[length(loss)])
  printf('loss calculated training: %f \n', NLL_train)
  
  printf('loss loaded validation:     %f \n', loss_val[length(loss_val)])
  printf('loss calculated validation: %f \n', NLL_val)
} 

######## Simple Stuff ######

#' creates a filename for a figure by adding SUFFIX and place of DB
#'
#' @param name 
#'
#' @return
make_fn = function(name) {
  return(paste0(DROPBOX, "figures/", SUFFIX,"_",name)) 
}




























