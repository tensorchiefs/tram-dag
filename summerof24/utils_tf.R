# bern_utils #####
R_START = 1-0.0001 #1.0-1E-1
L_START = 0.0001

# The following code has been partly (stuff which does not need TFP) been taken from utils.R

##################################
##### Utils for tram_dag #########
version_info = function(){
  print(reticulate::py_config())
  print('R version:')
  print('tensorflow version:')
  print(tf$version$VERSION)
  print('keras version:')
  print(reticulate::py_run_string("import keras; print(keras.__version__)"))
  #print('tfprobability version:')
  #print(tfprobability::tfp_version())
}

# Below is for creating documentation
if (FALSE){
  roxygen2::roxygenise("R/tram_dag/")
  devtools::document("R/tram_dag/")
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


## Training in which the beta layers are trained with a 2nd order update using a Hessian
train_step_with_hessian_beta <- tf_function(autograph = TRUE, 
                                            function(train_data, beta_weights, optimizer, lr_hessian = 0.1 ) {
#train_step <- function(train_data, beta_weights) {
    # train_data = train$df_orig
    #with(tf$GradientTape(persistent = TRUE) %as% tape2, { # Gradients for second-order derivatives
    #  with(tf$GradientTape(persistent = TRUE) %as% tape1, { # Gradients for first-order derivatives 
    with(tf$GradientTape() %as% tape2, { # Gradients for second-order derivatives
      with(tf$GradientTape() %as% tape1, { # Gradients for first-order derivatives 
        h_params <- param_model(train_data)
        loss <- struct_dag_loss(train_data, h_params)
        #hist = param_model$fit(x = train$df_orig, y=train$df_orig, epochs = 500L,verbose = TRUE)
      })
    
      # Compute first-order gradients
      all_weights <- param_model$trainable_weights
      all_grads <- tape1$gradient(loss, all_weights)
      #optimizer$apply_gradients(purrr::transpose(list(all_grads, all_weights))) #HACKATTACK
      other_gradients <- all_grads[!sapply(param_model$trainable_weights, function(weight) {
        identical(weight$name, beta_weights$name)
      })]
      beta_gradients <- all_grads[sapply(param_model$trainable_weights, function(weight) {
        identical(weight$name, beta_weights$name)
      })]
      other_weights <- param_model$trainable_weights[!sapply(param_model$trainable_weights, function(weight) {
        identical(weight$name, beta_weights$name)
      })]
      if (length(beta_gradients) != 1) {
        stop("Current implementation only supports **one** beta layer")
      }
      b = beta_gradients[[1]]
      bl_shape <- beta_weights$shape
      hessians <- tape2$jacobian(beta_gradients[[1]], beta_weights)  
      
   }) 
  optimizer$apply_gradients(purrr::transpose(list(other_gradients, other_weights))) 
  # Manipulate gradients and apply them 
    # Flatten the Hessian tensor to a matrix for inversion
    hessian_size <- bl_shape[[1]] * bl_shape[[2]]
    hessian_flat <- tf$reshape(hessians, shape = c(hessian_size, hessian_size))  # Adjust shape as needed
    # Add regularization to the Hessian
    hessian_flat <- hessian_flat + tf$eye(hessian_size) * 1e-8
    # Compute the inverse of the Hessian matrix
    hessian_inv <- tf$linalg$inv(hessian_flat)
    # DEBUG HACK ATTACK - replace with tf$linalg$inv when fixed <-------------TODO 
    #hessian_inv <- tf$eye(hessian_size)  # Identity matrix of appropriate size
    # Flatten the gradient for matrix multiplication
    grads_flat <- tf$reshape(beta_gradients[[1]], shape = c(hessian_size, 1L))
    # Compute the update using Hessian and gradient (this is the newton update rule)
    beta_update <- tf$matmul(hessian_inv, grads_flat)
    # Reshape the update back to the original shape
    beta_update_reshaped <- tf$reshape(beta_update, shape = bl_shape)  # Adjust shape as needed
    # Apply the update to the beta weights with the learning rate
    beta_weights$assign_sub(lr_hessian * beta_update_reshaped)
  loss
})  


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
  if (target_index != NULL) {
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
    NLL_train = NLL_train + calc_NLL(thetaNN_l[[i]], train_data$parents[[i]], train_data$target[[i]])$numpy()
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

# utils_dag_maf (Taken from) #####
# The following code has been completely taken from utils_dag_maf.R
###### Builds a model #####
create_theta_tilde_maf = function(adjacency, len_theta, layer_sizes){
  input_layer <- layer_input(shape = list(ncol(adjacency)))
  outs = list()
  for (r in 1:len_theta){
    d = input_layer
    for (i in 2:(length(layer_sizes) - 1)) {
      d = LinearMasked(units=layer_sizes[i], mask=t(masks[[i-1]]))(d)
      #d = layer_activation(activation='relu')(d)
      d = layer_activation(activation='sigmoid')(d)
    }
    out = LinearMasked(units=layer_sizes[length(layer_sizes)], mask=t(masks[[length(layer_sizes) - 1]]))(d)
    outs = append(outs,tf$expand_dims(out, axis=-1L)) #Expand last dim for concatenating
  }
  outs_c = keras$layers$concatenate(outs, axis=-1L)
  model = keras_model(inputs = input_layer, outputs = outs_c)
  return(model)
}

create_param_net <- function(len_param, input_layer, layer_sizes, masks, last_layer_bias=TRUE) {
  outs = list()
  for (r in 1:len_param){
    d = input_layer
    if (length(layer_sizes) > 2){ #Hidden Layers
      for (i in 2:(length(layer_sizes) - 1)) {
        d = LinearMasked(units=layer_sizes[i], mask=t(masks[[i-1]]))(d)
        #d = layer_activation(activation='relu')(d)
        d = layer_activation(activation='sigmoid')(d)
      }
    } #add output layers
    out = LinearMasked(units=layer_sizes[length(layer_sizes)], mask=t(masks[[length(layer_sizes) - 1]]),bias=last_layer_bias)(d)
    outs = append(outs,tf$expand_dims(out, axis=-1L)) #Expand last dim for concatenating
  }
  outs_c = keras$layers$concatenate(outs, axis=-1L)
}

# Creates a keras layer which takes as input (None, |x|) and returns (None, |x|, 1) which are all zero 
create_null_net <- function(input_layer) {
  output_layer <- layer_lambda(input_layer, function(x) {
    # Create a tensor of zeros with the same shape as x
    zeros_like_x <- k_zeros_like(x)
    # Add an extra dimension to match the desired output shape (None, |x|, 1)
    expanded_zeros_like_x <- k_expand_dims(zeros_like_x, -1)
    return(expanded_zeros_like_x)
  })
  return(output_layer)
}

# Creates a keras layer which takes as input (None, |x|) and returns (None, |x|, len) which are all constant variables


create_param_model = function(MA, hidden_features_I = c(2,2), len_theta=30, hidden_features_CS = c(2,2)){
  input_layer <- layer_input(shape = list(ncol(MA)))
  
  ##### Creating the Intercept Model
  if ('ci' %in% MA == TRUE) { # At least one 'ci' in model
    layer_sizes_I <- c(ncol(MA), hidden_features_I, nrow(MA))
    masks_I = create_masks(adjacency =  t(MA == 'ci'), hidden_features_I)
    h_I = create_param_net(len_param = len_theta, input_layer=input_layer, layer_sizes = layer_sizes_I, masks_I, last_layer_bias=TRUE)
    #dag_maf_plot(masks_I, layer_sizes_I)
    #model_ci = keras_model(inputs = input_layer, h_I)
  } else { # Adding simple interceps
    layer_sizes_I = c(ncol(MA), nrow(MA))
    masks_I = list(matrix(FALSE, nrow=nrow(MA), ncol=ncol(MA)))
    h_I = create_param_net(len_param = len_theta, input_layer=input_layer, layer_sizes = layer_sizes_I, masks_I, last_layer_bias=TRUE)
  }
  
  ##### Creating the Complex Shift Model
  if ('cs' %in% MA == TRUE) { # At least one 'cs' in model
    layer_sizes_CS <- c(ncol(MA), hidden_features_CS, nrow(MA))
    masks_CS = create_masks(adjacency =  t(MA == 'cs'), hidden_features_CS)
    h_CS = create_param_net(len_param = 1, input_layer=input_layer, layer_sizes = layer_sizes_CS, masks_CS, last_layer_bias=FALSE)
    #dag_maf_plot(masks_CS, layer_sizes_CS)
    # model_cs = keras_model(inputs = input_layer, h_CS)
  } else { #No 'cs' term in model --> return zero
    h_CS = create_null_net(input_layer)
  }
  
  ##### Creating the Linear Shift Model
  if ('ls' %in% MA == TRUE) {
    #h_LS = keras::layer_dense(input_layer, use_bias = FALSE, units = 1L)
    layer_sizes_LS <- c(ncol(MA), nrow(MA))
    masks_LS = create_masks(adjacency =  t(MA == 'ls'), c())
    out = LinearMasked(units=layer_sizes_LS[2], mask=t(masks_LS[[1]]), bias=FALSE, name='beta')(input_layer) 
    h_LS = tf$expand_dims(out, axis=-1L)#keras$layers$concatenate(outs, axis=-1L)
    #dag_maf_plot(masks_LS, layer_sizes_LS)
    #model_ls = keras_model(inputs = input_layer, h_LS)
  } else {
    h_LS = create_null_net(input_layer)
  }
  #Keras does not work with lists (only in eager mode)
  #model = keras_model(inputs = input_layer, outputs = list(h_I, h_CS, h_LS))
  #Dimensions h_I (B,3,30) h_CS (B, 3, 1) h_LS(B, 3, 3)
  # Convention for stacking
  # 1       CS
  # 2->|X|+1 LS
  # |X|+2 --> Ende M 
  outputs_tensor = keras$layers$concatenate(list(h_CS, h_LS, h_I), axis=-1L)
  param_model = keras_model(inputs = input_layer, outputs = outputs_tensor)
  return(param_model)
}


###### to_theta3 ####
# See zuko but fixed for order 3
to_theta3 = function(theta_tilde){
  shift = tf$convert_to_tensor(log(2) * dim(theta_tilde)[[length(dim(theta_tilde))]] / 2)
  order = tf$shape(theta_tilde)[3]
  widths = tf$math$softplus(theta_tilde[,, 2L:order, drop=FALSE])
  widths = tf$concat(list(theta_tilde[,, 1L, drop=FALSE], widths), axis = -1L)
  return(tf$cumsum(widths, axis = -1L) - shift)
}

### Bernstein Basis Polynoms of order M (i.e. M+1 coefficients)
# return (B,Nodes,M+1)
bernstein_basis <- function(tensor, M) {
  # Ensure tensor is a TensorFlow tensor
  tensor <- tf$convert_to_tensor(tensor)
  dtype <- tensor$dtype
  M = tf$cast(M, dtype)
  # Expand dimensions to allow broadcasting
  tensor_expanded <- tf$expand_dims(tensor, -1L)
  # Ensuring tensor_expanded is within the range (0,1) 
  tensor_expanded = tf$clip_by_value(tensor_expanded, tf$keras$backend$epsilon(), 1 - tf$keras$backend$epsilon())
  k_values <- tf$range(M + 1L) #from 0 to M
  
  # Calculate the Bernstein basis polynomials
  log_binomial_coeff <- tf$math$lgamma(M + 1.) - 
    tf$math$lgamma(k_values + 1.) - 
    tf$math$lgamma(M - k_values + 1.)
  log_powers <- k_values * tf$math$log(tensor_expanded) + 
    (M - k_values) * tf$math$log(1 - tensor_expanded)
  log_bernstein <- log_binomial_coeff + log_powers
  
  return(tf$exp(log_bernstein))
}


###### LinearMasked ####
LinearMasked(keras$layers$Layer) %py_class% {
  
  initialize <- function(units = 32, mask = NULL, bias=TRUE, name = NULL, trainable = NULL, dtype = NULL) {
    super$initialize(name = name)
    self$units <- units
    self$mask <- mask  # Add a mask parameter
    self$bias = bias
    # The additional arguments (name, trainable, dtype) are not used but are accepted to prevent errors during deserialization
  }
  
  build <- function(input_shape) {
    self$w <- self$add_weight(
      name = "w",
      shape = shape(input_shape[[2]], self$units),
      initializer = "random_normal",
      trainable = TRUE
    )
    if (self$bias) {
      self$b <- self$add_weight(
        name = "b",
        shape = shape(self$units),
        initializer = "random_normal",
        trainable = TRUE
      )
    } else{
      self$b <- NULL
    }
    
    # Handle the mask conversion if it's a dictionary (when loaded from a saved model)
    if (!is.null(self$mask)) {
      np <- import('numpy')
      if (is.list(self$mask) || "AutoTrackable" %in% class(self$mask)) {
        # Extract the mask value and dtype from the dictionary
        mask_value <- self$mask$config$value
        mask_dtype <- self$mask$config$dtype
        print("Hallo Gallo")
        mask_dtype = 'float32'
        print(mask_dtype)
        # Convert the mask value back to a numpy array
        mask_np <- np$array(mask_value, dtype = mask_dtype)
        # Convert the numpy array to a TensorFlow tensor
        self$mask <- tf$convert_to_tensor(mask_np, dtype = mask_dtype)
      } else {
        # Ensure the mask is the correct shape and convert it to a tensor
        if (!identical(dim(self$mask), dim(self$w))) {
          stop("Mask shape must match weights shape")
        }
        self$mask <- tf$convert_to_tensor(self$mask, dtype = self$w$dtype)
      }
    }
  }
  
  call <- function(inputs) {
    if (!is.null(self$mask)) {
      # Apply the mask
      masked_w <- self$w * self$mask
    } else {
      masked_w <- self$w
    }
    if(!is.null(self$b)){
      tf$matmul(inputs, masked_w) + self$b
    } else{
      tf$matmul(inputs, masked_w)
    }
  }
  
  get_config <- function() {
    config <- super$get_config()
    config$units <- self$units
    config$mask <- if (!is.null(self$mask)) tf$make_ndarray(tf$make_tensor_proto(self$mask)) else NULL
    config
  }
}


###### Pure R Function ####
# Creates Autoregressive masks for a given adjency matrix and hidden features
create_masks <- function(adjacency, hidden_features=c(64, 64)) {
  out_features <- nrow(adjacency)
  in_features <- ncol(adjacency)
  
  #adjacency_unique <- unique(adjacency, MARGIN = 1)
  #inverse_indices <- match(as.matrix(adjacency), as.matrix(adjacency_unique))
  
  #np.dot(adjacency.astype(int), adjacency.T.astype(int)) == adjacency.sum(axis=-1, keepdims=True).T
  d = tcrossprod(adjacency * 1L)
  precedence <-  d == matrix(rowSums(adjacency * 1L), ncol=nrow(adjacency), nrow=nrow(d), byrow = TRUE)
  
  masks <- list()
  for (i in seq_along(c(hidden_features, out_features))) {
    if (i > 1) {
      mask <- precedence[, indices, drop = FALSE]
    } else {
      mask <- adjacency
    }
    
    if (all(!mask)) {
      stop("The adjacency matrix leads to a null Jacobian.")
    }
    
    if (i <= length(hidden_features)) {
      reachable <- which(rowSums(mask) > 0)
      if (length(reachable) > 0) {
        indices <- reachable[(seq_len(hidden_features[i]) - 1) %% length(reachable) + 1]
      } else {
        indices <- integer(0)
      }
      mask <- mask[indices, , drop = FALSE]
    } 
    #else {
    #  mask <- mask[inverse_indices, , drop = FALSE]
    #}
    masks[[i]] <- mask
  }
  return(masks)
}

########## Transformations ############
check_baselinetrafo = function(h_params){
  #h_params = param_model(train$df_orig)
  k_min <- k_constant(global_min)
  k_max <- k_constant(global_max)
  
  k_min.df <- k_min$numpy()
  k_max.df <- k_max$numpy()
  
  theta_tilde <- h_params[,,3:dim(h_params)[3], drop = FALSE]
  theta = to_theta3(theta_tilde)
  length.out  = nrow(theta_tilde)
  X = matrix(NA, nrow=length.out, ncol=length(k_min))
  for(i in 1:length(k_min.df)){
    X[,i] = seq(k_min.df[i], k_max.df[i], length.out = length.out)
  }
  t_i = k_constant(X)
  
  h_I = h_dag_extra(t_i, theta, k_min, k_max) 
  return(list(h_I = h_I$numpy(), Xs=X))
} 



sample_standard_logistic <- function(shape, epsilon=1e-7) {
  uniform_samples <- tf$random$uniform(shape, minval=0, maxval=1)
  clipped_uniform_samples <- tf$clip_by_value(uniform_samples, epsilon, 1 - epsilon)
  logistic_samples <- tf$math$log(clipped_uniform_samples / (1 - clipped_uniform_samples))
  return(logistic_samples)
}


### h_dag
h_dag = function(t_i, theta){
  len_theta = tf$shape(theta)[3L] #TODO tied to 3er Tensors
  Be = bernstein_basis(t_i, len_theta-1L) 
  return (tf$reduce_mean(theta * Be, -1L))
}

### h_dag_dash
h_dag_dash = function(t_i, theta){
  len_theta = tf$shape(theta)[3L] #TODO tied to 3er Tensors
  Bed = bernstein_basis(t_i, len_theta-2L) 
  dtheta = (theta[,,2:len_theta,drop=FALSE]-theta[,,1:(len_theta-1L), drop=FALSE])
  return (tf$reduce_sum(dtheta * Bed, -1L))
}


h_dag_extra = function(t_i, theta, k_min, k_max){
  DEBUG = FALSE
  t_i = (t_i - k_min)/(k_max - k_min) # Scaling
  t_i3 = tf$expand_dims(t_i, axis=-1L)
  # for t_i < 0 extrapolate with tangent at h(0)
  b0 <- tf$expand_dims(h_dag(L_START, theta),axis=-1L)
  slope0 <- tf$expand_dims(h_dag_dash(L_START, theta), axis=-1L) 
  # If t_i < 0, use a linear extrapolation
  mask0 <- tf$math$less(t_i3, L_START)
  h <- tf$where(mask0, slope0 * (t_i3 - L_START) + b0, t_i3)
  #if (DEBUG) printf('~~~ eval_h_extra  Fraction of extrapolated samples < 0 : %f \n', tf$reduce_mean(tf$cast(mask0, tf$float32)))
  
  #(for t_i > 1)
  b1 <- tf$expand_dims(h_dag(R_START, theta),axis=-1L)
  slope1 <-  tf$expand_dims(h_dag_dash(R_START, theta), axis=-1L)
  # If t_i > 1, use a linear extrapolation
  mask1 <- tf$math$greater(t_i3, R_START)
  h <- tf$where(mask1, slope1 * (t_i3 - R_START) + b1, h)
  if (DEBUG) printf('~~~ eval_h_extra  Fraction of extrapolated samples > 1 : %f \n', tf$reduce_mean(tf$cast(mask1, tf$float32)))
  
  # For values in between, use the original function
  mask <- tf$math$logical_and(tf$math$greater_equal(t_i3, L_START), tf$math$less_equal(t_i3, R_START))
  h <- tf$where(mask, tf$expand_dims(h_dag(t_i, theta), axis=-1L), h)
  # Return the mean value
  return(tf$squeeze(h))
}

h_dag_extra_struc = function(t_i, theta, shift, k_min, k_max){
  #Throw unsupported error
  DEBUG = FALSE
  #stop('Please check before removing')
  #k_min <- k_constant(global_min)
  #k_max <- k_constant(global_max)
  t_i = (t_i - k_min)/(k_max - k_min) # Scaling
  t_i3 = tf$expand_dims(t_i, axis=-1L)
  # if (length(t_i$shape) == 2) {
  #   t_i3 = tf$expand_dims(t_i, axis=-1L)
  # } 
  # for t_i < 0 extrapolate with tangent at h(0)
  b0 <- tf$expand_dims(h_dag(L_START, theta) + shift,axis=-1L)
  slope0 <- tf$expand_dims(h_dag_dash(L_START, theta), axis=-1L) 
  # If t_i < 0, use a linear extrapolation
  mask0 <- tf$math$less(t_i3, L_START)
  h <- tf$where(mask0, slope0 * (t_i3 - L_START) + b0, t_i3)
  #if (DEBUG) printf('~~~ eval_h_extra  Fraction of extrapolated samples < 0 : %f \n', tf$reduce_mean(tf$cast(mask0, tf$float32)))
  
  #(for t_i > 1)
  b1 <- tf$expand_dims(h_dag(R_START, theta) + shift,axis=-1L)
  slope1 <-  tf$expand_dims(h_dag_dash(R_START, theta), axis=-1L)
  # If t_i > 1, use a linear extrapolation
  mask1 <- tf$math$greater(t_i3, R_START)
  h <- tf$where(mask1, slope1 * (t_i3 - R_START) + b1, h)
  if (DEBUG) printf('~~~ eval_h_extra  Fraction of extrapolated samples > 1 : %f \n', tf$reduce_mean(tf$cast(mask1, tf$float32)))
  
  # For values in between, use the original function
  mask <- tf$math$logical_and(tf$math$greater_equal(t_i3, L_START), tf$math$less_equal(t_i3, R_START))
  h <- tf$where(mask, tf$expand_dims(h_dag(t_i, theta) + shift, axis=-1L), h)
  # Return the mean value
  return(tf$squeeze(h))
}

h_dag_dash_extra = function(t_i, theta, k_min, k_max){
  t_i = (t_i - k_min)/(k_max - k_min) # Scaling
  t_i3 = tf$expand_dims(t_i, axis=-1L)
  
  #Left extrapolation
  slope0 <- tf$expand_dims(h_dag_dash(L_START, theta), axis=-1L) 
  mask0 <- tf$math$less(t_i3, L_START)
  h_dash <- tf$where(mask0, slope0, t_i3)
  
  #Right extrapolation
  slope1 <-  tf$expand_dims(h_dag_dash(R_START, theta), axis=-1L)
  mask1 <- tf$math$greater(t_i3, R_START)
  h_dash <- tf$where(mask1, slope1, h_dash)
  
  #Interpolation
  mask <- tf$math$logical_and(tf$math$greater_equal(t_i3, L_START), tf$math$less_equal(t_i3, R_START))
  h_dash <- tf$where(mask, tf$expand_dims(h_dag_dash(t_i,theta),axis=-1L), h_dash)
  
  return (tf$squeeze(h_dash))
}

dag_loss = function (t_i, theta_tilde){
  theta = to_theta3(theta_tilde)
  h_ti = h_dag_extra(t_i, theta)
  # The log of the logistic density at h is log(f(h))=−h−2log(1+e −h)
  # log_density2 = -h_ti - 2 * tf$math$log(1 + tf$math$exp(-h_ti))
  # Softpuls is nuerically more stable (according to ChatGPT) compared to log_density2
  log_latent_density = -h_ti - 2 * tf$math$softplus(-h_ti)
  h_dag_dashd = h_dag_dash_extra(t_i, theta)
  log_lik = log_latent_density + tf$math$log(tf$math$abs(h_dag_dashd))
  return (-tf$reduce_mean(log_lik))#(-tf$reduce_mean(log_lik, axis=-1L))
}

# Define the function to calculate the logistic CDF
logistic_cdf <- function(x) {
  return(tf$math$reciprocal(tf$math$add(1, tf$math$exp(-x))))
}

struct_dag_loss = function (t_i, h_params){
  #t_i = train$df_orig
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
  #Thetas for intercept
  theta = to_theta3(theta_tilde)
  
  if (!exists('data_type')){ #Defaulting to all continuous 
    cont_dims = 1:dim(theta_tilde)[2]
    cont_ord = c()
  } else{ 
    cont_dims = which(data_type == 'c')
    cont_ord = which(data_type == 'o')
  }
  if (len_theta == -1){ 
    len_theta = dim(theta_tilde)[3]
  }
  
  NLL = 0
  ### Continiuous dimensions
  #### At least one continuous dimension exits
  if (length(cont_dims) != 0){
    h_I = h_dag_extra(t_i[,cont_dims, drop=FALSE], theta[,cont_dims,1:len_theta,drop=FALSE], k_min[cont_dims], k_max[cont_dims]) 
    h = h_I + h_LS[,cont_dims, drop=FALSE] + h_CS[,cont_dims, drop=FALSE]
    
    #Compute terms for change of variable formula
    log_latent_density = -h - 2 * tf$math$softplus(-h) #log of logistic density at h
    ## h' dh/dtarget is 0 for all shift terms
    log_hdash = tf$math$log(tf$math$abs(
      h_dag_dash_extra(t_i[,cont_dims, drop=FALSE], theta[,cont_dims,1:len_theta,drop=FALSE], k_min[cont_dims], k_max[cont_dims]))
      ) - 
      tf$math$log(k_max[cont_dims] - k_min[cont_dims])  #Chain rule! See Hathorn page 12 
    
    NLL = NLL - tf$reduce_mean(log_latent_density + log_hdash)
  }
  
  ### Ordinal dimensions
  if (length(cont_ord) != 0){
    B = tf$shape(t_i)[1]
    for (col in cont_ord){
      nol = tf$cast(k_max[col] - 1L, tf$int32) # Number of cut-points in respective dimension
      theta_ord = theta[,col,1:nol,drop=TRUE] # Intercept


      h = theta_ord + h_LS[,col, drop=FALSE] + h_CS[,col, drop=FALSE]
      # putting -Inf and +Inf to the left and right of the cutpoints
      neg_inf = tf$fill(c(B,1L), -Inf)
      pos_inf = tf$fill(c(B,1L), +Inf)
      h_with_inf = tf$concat(list(neg_inf, h, pos_inf), axis=-1L)
      logistic_cdf_values = logistic_cdf(h_with_inf)
      #cdf_diffs <- tf$subtract(logistic_cdf_values[, 2:ncol(logistic_cdf_values)], logistic_cdf_values[, 1:(ncol(logistic_cdf_values) - 1)])
      cdf_diffs <- tf$subtract(logistic_cdf_values[, 2:tf$shape(logistic_cdf_values)[2]], logistic_cdf_values[, 1:(tf$shape(logistic_cdf_values)[2] - 1)])
      # Picking the observed cdf_diff entry
      class_indices <- tf$cast(t_i[, col] - 1, tf$int32)  # Convert to zero-based index
      # Create batch indices to pair with class indices
      batch_indices <- tf$range(tf$shape(class_indices)[1])
      # Combine batch_indices and class_indices into pairs of indices
      gather_indices <- tf$stack(list(batch_indices, class_indices), axis=1)
      cdf_diff_picked <- tf$gather_nd(cdf_diffs, gather_indices)
      # Gather the corresponding values from cdf_diffs
      NLL = NLL -tf$reduce_mean(tf$math$log(cdf_diff_picked))
    }
  }
  
  ### DEBUG 
  #if (sum(is.infinite(log_lik$numpy())) > 0){
  #  print("Hall")
  #}
  return (NLL)
}

# Old version of the struct_dag_loss used until 21 May 24 with Scalar Data
struct_dag_loss_OLD = function (t_i, h_params){
  # from the last dimension of h_params the first entriy is h_cs1
  # the second to |X|+1 are the LS
  # the 2+|X|+1 to the end is H_I
  h_cs <- h_params[,,1, drop = FALSE]
  h_ls <- h_params[,,2, drop = FALSE]
  theta_tilde <- h_params[,,3:dim(h_params)[3], drop = FALSE]
  #CI 
  theta = to_theta3(theta_tilde)
  h_I = h_dag_extra(t_i, theta)
  #LS
  h_LS = tf$squeeze(h_ls, axis=-1L)#tf$einsum('bx,bxx->bx', t_i, beta)
  #CS
  h_CS = tf$squeeze(h_cs, axis=-1L)
  
  h = h_I + h_LS + h_CS
  
  #Compute terms for change of variable formula
  log_latent_density = -h - 2 * tf$math$softplus(-h) #log of logistic density at h
  ## h' dh/dtarget is 0 for all shift terms
  log_hdash = tf$math$log(tf$math$abs(h_dag_dash_extra(t_i, theta)))
  
  log_lik = log_latent_density + log_hdash
  ### DEBUG 
  #if (sum(is.infinite(log_lik$numpy())) > 0){
  #  print("Hall")
  #}
  return (-tf$reduce_mean(log_lik))
}


dag_loss_dumm = function (t_i, theta_tilde){
  theta = to_theta3(theta_tilde)
  h_ti = h_dag_extra(t_i, theta)
  return (-tf$reduce_mean(h_ti, axis=-1L)) 
}

sample_logistics_within_bounds <- function(h_0, h_1) {
  samples <- map2_dbl(h_0, h_1, function(lower_bound, upper_bound) {
    while(TRUE) {
      sample <- as.numeric(tf$squeeze(sample_standard_logistic(c(1L,1L))))
      if (lower_bound < sample && sample < upper_bound) {
        return(sample)
      }
    }
  })
  return(samples)
}

# Load the required library
library(ggplot2)
library(grid)

# Function to draw the network
dag_maf_plot <- function(layer_masks, layer_sizes) {
  max_nodes <- max(layer_sizes)
  width <- max_nodes * 100
  min_x <- 0
  max_x <- width  # Adjust max_x to include input layer
  min_y <- Inf
  max_y <- -Inf
  
  # Create a data frame to store node coordinates
  nodes <- data.frame(x = numeric(0), y = numeric(0), label = character(0))
  
  # Draw the nodes for all layers
  for (i in 1:length(layer_sizes)) {
    size <- layer_sizes[i]
    layer_top <- max_nodes / 2 - size / 2
    
    for (j in 1:size) {
      x <- (i-1) * width
      y <- layer_top + j * 100
      label <- ifelse(i == 1, paste("x_", j, sep = ""), "")  # Add labels for the first column
      nodes <- rbind(nodes, data.frame(x = x, y = y, label=label))
      max_x <- max(max_x, x)
      min_y <- min(min_y, y)
      max_y <- max(max_y, y)
    }
  }
  
  # Create a data frame to store connection coordinates
  connections <- data.frame(x_start = numeric(0), y_start = numeric(0),
                            x_end = numeric(0), y_end = numeric(0))
  
  # Draw the connections
  for (i in 1:length(layer_masks)) {
    mask <- t(layer_masks[[i]])
    input_size <- nrow(mask)
    output_size <- ncol(mask)
    
    for (j in 1:input_size) {
      for (k in 1:output_size) {
        if (mask[j, k]) {
          start_x <- (i - 1) * width
          start_y <- max_nodes / 2 - input_size / 2 + j * 100
          end_x <- i * width
          end_y <- max_nodes / 2 - output_size / 2 + k * 100
          
          connections <- rbind(connections, data.frame(x_start = start_x, y_start = start_y,
                                                       x_end = end_x, y_end = end_y))
        }
      }
    }
  }
  
  
  # Create the ggplot object
  network_plot <- ggplot() +
    geom_segment(data = connections, aes(x = x_start, y = -y_start, xend = x_end, yend = -y_end),
                 color = 'black', size = 1,
                 arrow = arrow()) +
    geom_point(data = nodes, aes(x = x, y = -y), color = 'blue', size = 8,alpha = 0.5) +
    geom_text(data = nodes, aes(x = x, y = -y, label = label), vjust = 0, hjust = 0.5) +  # Add labels
    theme_void() 
  
  return(network_plot)
}




