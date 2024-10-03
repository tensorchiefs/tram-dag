R_START = 1-0.0001 #1.0-1E-1
L_START = 0.0001

utils_scale = function (y){
  min_y = min(y) 
  max_y = max(y) 
  return ( (y-min_y)/(max_y-min_y) )
}

utils_back_scale = function(y_scale, y){
  min_y = min(y)
  max_y = max(y)
  return (y_scale * (max_y - min_y) + min_y)
}

############################################################
# utils for bernstein
############################################################

# Construct h according to MLT paper (Hothorn, Möst, Bühlmann, p12)
init_beta_dist_for_h = function(len_theta){
  beta_dist = tfd_beta(1:len_theta, len_theta:1) 
  return (beta_dist)
}

# Construct h_dash according to MLT (Hothorn, Möst, Bühlmann, p12) correcting the factor M/(M+1) with 1 
init_beta_dist_for_h_dash = function(len_theta){
  M = len_theta - 1
  beta_dist = tfd_beta(1:M, M:1)
  return (beta_dist)
}

eval_h = function(theta_im, y_i, beta_dist_h){
  #if (DEBUG) print('EVAL H')
  y_i = tf$clip_by_value(y_i,L_START, R_START)
  f_im = beta_dist_h$prob(y_i) 
  return (tf$reduce_mean(f_im * theta_im, axis=1L))
}

eval_h_extra <- function(theta_im, y_i, beta_dist_h, beta_dist_h_dash) {
  
  # for y_i < 0 extrapolate with tangent at h(0)
  slope0 <- tf$expand_dims(eval_h_dash(theta_im, L_START, beta_dist_h_dash), axis=1L)
  b0 <- tf$expand_dims(eval_h(theta_im, L_START, beta_dist_h),1L)
  # If y_i < 0, use a linear extrapolation
  mask0 <- tf$math$less(y_i, L_START)
  h <- tf$where(mask0, slope0 * (y_i - L_START) + b0, y_i)
  if (DEBUG) printf('~~~ eval_h_extra  Fraction of extrapolated samples < 0 : %f \n', tf$reduce_mean(tf$cast(mask0, tf$float32)))
  
  #(for y_i > 1)
  b1 <- tf$expand_dims(eval_h(theta_im, R_START, beta_dist_h),1L)
  slope1 <- tf$expand_dims(eval_h_dash(theta_im, R_START, beta_dist_h_dash), axis=1L)
  # If y_i > 1, use a linear extrapolation
  mask1 <- tf$math$greater(y_i, R_START)
  h <- tf$where(mask1, slope1 * (y_i - R_START) + b1, h)
  if (DEBUG) printf('~~~ eval_h_extra  Fraction of extrapolated samples > 1 : %f \n', tf$reduce_mean(tf$cast(mask1, tf$float32)))
  
  # For values in between, use the original function
  mask <- tf$math$logical_and(tf$math$greater_equal(y_i, L_START), tf$math$less_equal(y_i, R_START))
  h <- tf$where(mask, beta_dist_h$prob(y_i)* theta_im, h)
  # Return the mean value
  return(tf$reduce_mean(h , axis=1L))
}

eval_h_dash_extra = function(theta_im, y_i, beta_dist_h_dash){
  
  # for y_i < 0 extrapolate with tangent at h(0)
  slope0 <- tf$expand_dims(eval_h_dash(theta_im, L_START, beta_dist_h_dash), axis=1L)
  # If y_i < 0, use a linear extrapolation
  mask0 <- tf$math$less(y_i, L_START)
  h_dash <- tf$where(mask0, slope0, y_i)
  if (DEBUG) printf('~~~ eval_h_dash_extra  Fraction of extrapolated samples < 0 : %f \n', tf$reduce_mean(tf$cast(mask0, tf$float32)))
  
  #(for y_i > 1)
  slope1 <- tf$expand_dims(eval_h_dash(theta_im, R_START, beta_dist_h_dash), axis=1L)
  # If y_i > 1, use a linear extrapolation
  mask1 <- tf$math$greater(y_i, R_START)
  h_dash <- tf$where(mask1, slope1, h_dash)
  if (DEBUG) printf('~~~ eval_h_extra  Fraction of extrapolated samples > 1 : %f \n', tf$reduce_mean(tf$cast(mask1, tf$float32)))
  
  
  # For values in between, use the original function
  mask <- tf$math$logical_and(tf$math$greater_equal(y_i, L_START), tf$math$less_equal(y_i, R_START))
  
  h_dash <- tf$where(mask, tf$expand_dims(eval_h_dash(theta_im, y_i, beta_dist_h_dash),1L), h_dash)
  return (tf$squeeze(h_dash))
}


eval_h_dash = function(theta, y, beta_dist_h_dash){
  y = tf$clip_by_value(y,L_START, R_START)
  by = beta_dist_h_dash$prob(y) 
  dtheta = (theta[,2:(ncol(theta))]-theta[,1:(ncol(theta)-1)])
  return (tf$reduce_sum(by * dtheta, axis=1L))
}



# ######## function that turns pre_theta from NN ouput to theta
to_theta = function(pre_theta){
  #Trivially calclutes theta_1 = h_1, theta_k = theta_k-1 + exp(h_k)
  d = tf$concat( c(tf$zeros(c(nrow(pre_theta),1L)),
                   tf$slice(pre_theta, begin = c(0L,0L), size = c(nrow(pre_theta),1L)),
                   #tf$math$exp(pre_theta[,2:ncol(pre_theta)])),axis=1L)
                   tf$math$softplus(pre_theta[,2:ncol(pre_theta)])),axis=1L)
  #python d = tf.concat( (tf.zeros((h.shape[0],1)), h[:,0:1], tf.math.softplus(h[:,1:h.shape[1]])),axis=1)
  return (tf$cumsum(d[,2L:ncol(d)], axis=1L))
}

########################
# Class bernp
# S3 Class build according to https://adv-r.hadley.nz/s3.html
# My first class in R, hence some comments

#"Private" and complete constructor. This constructor verifies the input.
new_bernp = function(len_theta = integer()){
  stopifnot(is.integer(len_theta))
  stopifnot(len_theta > 0)
  structure( #strcutur is a bunch of data with 
    list( #bunch of data
      len_theta=len_theta,  
      beta_dist_h = init_beta_dist_for_h(len_theta),
      beta_dist_h_dash = init_beta_dist_for_h_dash(len_theta),
      stdnorm = tfd_normal(loc=0, scale=1)
    ),
    class = "bernp" #class attribute set so it's a class
  )  
}

#"Public" Constructor to create an instance of class bernp
bernp = function(len_theta=integer()){
  new_bernp(as.integer(len_theta))
}

# Computes the trafo h_y(y|x) out_bern is the (unconstrained) output of the NN 
bernp.eval_h = function(bernp, out_bern, y){
  theta_im = to_theta(out_bern)
  eval_h(theta_im, y_i = y, beta_dist_h = bernp$beta_dist_h)
}

# Computes the trafo h'_y(y|x) out_bern is the (unconstrained) output of the NN 
bernp.eval_h_dash = function(bernp, out_bern, y){
  theta_im = to_theta(out_bern)
  eval_h_dash(theta_im, y, beta_dist_h = bernp$beta_dist_h_dash)
}


# Computs NLL out_bern is the (unconstrained ) output of the NN 
bernp.nll = function(bernp, out_bern, y, y_range=1, out_eta = NULL) {
  theta_im = to_theta(out_bern)
  if (is.null(out_eta)){
    z = eval_h(theta_im, y_i = y, beta_dist_h = bernp$beta_dist_h)
  }else{
    hy = eval_h(theta_im, y_i = y, beta_dist_h = bernp$beta_dist_h)
    z = hy - out_eta[,1]
  }
  h_y_dash = eval_h_dash(theta_im, y, beta_dist_h_dash = bernp$beta_dist_h_dash)
  return(-tf$math$reduce_mean(bernp$stdnorm$log_prob(z) + tf$math$log(h_y_dash)) + log(y_range) )
}

# Computs CPD for one row in the output batch  
bernp.p_y_h = function(bernp, out_row, from, to, length.out, out_eta = NULL){
  stopifnot(out_row$shape[start_index] == 1) #We need a single row
  theta_rep = to_theta(k_tile(out_row, c(length.out, 1)))
  y_cont = keras_array(matrix(seq(from,to,length.out = length.out), nrow=length.out,ncol=1))
  if (is.null(out_eta)){
    z = eval_h(theta_rep, y_cont, beta_dist_h = bernp$beta_dist_h)  
  } else{
    hy = eval_h(theta_rep, y_cont, beta_dist_h = bernp$beta_dist_h)  
    z = hy - out_eta[,1]
  }
  p_y = bernp$stdnorm$prob(z) * as.array(eval_h_dash(theta_rep, y_cont, beta_dist_h_dash = bernp$beta_dist_h_dash))
  df = data.frame(
    y = seq(from,to,length.out = length.out),
    p_y = p_y$numpy(),
    h = z$numpy()
  )
  return (df)
}


# eval_h = function(bernp, theta_im, y){
#    UseMethod("eval_h")
# }


# Implementation of generic methods
# print.bernp = function(x) {
#   print('Hallo print TODO ')
#   print(x$len_theta)
# }

# Implementation of novel generic methods
# p_y.bernp = function(bernp, out) {
#   print(paste0('Hallo p_y TODO ', bernp))
# }

# # Registration of new generics
# p_y = function(x, out) {
#   UseMethod("p_y")
# }


# # testing:
# #############################################
# # plain R functions from mlt_utils.r
# theta = matrix(c(
#   0.1,1.1,4,12,
#   2,3,4,5,
#   6,7,8,9,
#   10,11,12,13,
#   14,15,16,17
# ), nrow = 5, ncol = 4, byrow = TRUE)
# 
# y = array(c(0.00,0.25,0.50,0.75,1.00), c(5,1))
# theta[2,]
# 
# # use plain R fct h, h_dash with expect vectors for theta and y
# source("mlt_utils.r")
# ( y[2] )
# ( h(theta[2,], y[2]) )
# ( h_dash(theta[2,], y[2]) )
# 
# # reproduce result from plain R fct h and h_dash
# len_theta = 4
# 
# beta_dist_h = init_beta_dist_for_h(len_theta)
# beta_dist_h_dash = init_beta_dist_for_h_dash(len_theta)
# 
# # select one col from theta and one value from y and turn it into a tf tensor
# ( theta_select=tf$Variable( theta[2,,drop=FALSE], dtype='float32')  )
# ( y_select=tf$Variable( y[2,drop=FALSE], dtype='float32') )
# eval_h(theta_select,y_select)
# eval_h_dash(theta_select,y_select)
# 
# # let's do it for a whole tensor, like we get it dl applications
# ( y_dl = tf$Variable(y, dtype='float32') )
# ( theta_dl = tf$Variable(theta, dtype='float32') )
# 
# eval_h(theta_dl,y_dl)
# eval_h_dash(theta_dl,y_dl)
# 

###### test of to_theta function
##################################################
# 
# pre_theta_tmp = array(1:-20,dim=c(5,4))
# ( pre_theta_tmp = tf$Variable(pre_theta_tmp, dtype='float32') )
# to_theta(pre_theta_tmp)
# 
# # #Hier weiter machen - juhu!
# #
# #
# # ############
# # # Boston
# # data("BostonHousing2", package = "mlbench")
# # dat=BostonHousing2
# # source("mlt_utils.R")
# # y = matrix(utils_scale(dat$cmedv),ncol=1)
# # y = tf$Variable(y, dtype='float32')
# #
# #
# #
# #
# #
# #
# #
# #
#devtools::install_github("rstudio/keras")
#library(keras)
#install_keras()
#library(keras)
#install.packages("tfprobability")
# library(tfprobability)
# 
# library(tensorflow)
# # tf_version()
# # tf$version
# tf_probability()
# tf$compat$v2$enable_v2_behavior()



if (FALSE){
  DEBUG = TRUE
  library(R.utils)
  #print(d)
  #out = 4 #TODO replace if the output of the network
  #p_y(d, 4)
  #d$len_theta
  #d$beta_dist_h
  library(tfprobability)
  
  d = bernp(4)
  # Test input data
  ys = seq(-1,2,0.01)
  h = NA*ys
  hextra = h 
  hdash = h
  for (i in 1:length(ys)){
    theta_im <- tf$constant(c(-0.1, 12.2, 22.34, 122.4), shape=c(1L,4L))
    y_i <- tf$constant(ys[i], shape=c(1L,1L))
    # Call eval_h_inter function
    h[i] = eval_h(theta_im, y_i, d$beta_dist_h)$numpy()
    hdash[i] = eval_h_dash(theta_im, y_i, d$beta_dist_h_dash)$numpy()
    hextra[i]=eval_h_extra(theta_im, y_i, d$beta_dist_h, d$beta_dist_h_dash)$numpy()
  }
  plot(ys, hextra, col='red', lty=2, type='l')
  lines(ys, h,  type='l')
  #lines(ys, hdash, col='pink', lty=2)
  
  d$beta_dist_h$prob(y_i)
  eval_h
  
  y_i <- tf$constant(c(0.1,1.01), shape=c(2L,1L))
  eval_h(theta_im, y_i, d$beta_dist_h)
  eval_h_extra(theta_im, y_i, d$beta_dist_h, d$beta_dist_h_dash)
  
}


