windows = TRUE
if(windows){
  start_index=1
} else {
  start_index=0
}


# Requires bern_utils
make_bernp = function(len_theta){
  try( #Hack Attack
    {
      bernp(len_theta = len_theta)
      bernp(len_theta = len_theta)
    }
  )
  return(bernp(len_theta = len_theta))
}

make_hist = function(){
  # prepare data.frrame where we collect results in each step
  history = data.frame(matrix(NA, nrow = 1, ncol=4))
  names(history) = c('step', 'fold', 'nll_train', 'nll_test')
  history$method = 'NA'
  history.row = 1
  return (history)
}

make_hist_grid = function(){
  # prepare data.frrame where we collect results in each step
  history = data.frame(matrix(NA, nrow = 1, ncol=4))
  names(history) = c('step', 'fold', 'nll_train', 'nll_test')
  history$method = 'NA'
  history$regularization = 'NA'
  history$spatz = 'NA'
  history$x_scale = 'NA'
  history.row = 1
  return (history)
}

