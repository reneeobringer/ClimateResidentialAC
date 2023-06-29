# custom yardstick metric (for use in tidymodels)

library(yardstick)
library(rlang)

nrmse_vec <- function(truth, estimate, na_rm = TRUE, ...) {
  
  nrmse_impl <- function(truth, estimate) {
    (sqrt(mean((truth - estimate) ^ 2)))/(max(truth)-min(truth))
  }
  
  metric_vec_template(
    metric_impl = nrmse_impl,
    truth = truth, 
    estimate = estimate,
    na_rm = na_rm,
    cls = "numeric",
    ...
  )
  
}

data("solubility_test")

nrmse_vec(
  truth = solubility_test$solubility, 
  estimate = solubility_test$prediction
)

nrmse <- function(data, ...) {
  UseMethod("nrmse")
}

nrmse <- new_numeric_metric(nrmse, direction = "minimize")

nrmse.data.frame <- function(data, truth, estimate, na_rm = TRUE, ...) {
  
  metric_summarizer(
    metric_nm = "nrmse",
    metric_fn = nrmse_vec,
    data = data,
    truth = !! enquo(truth),
    estimate = !! enquo(estimate), 
    na_rm = na_rm,
    ...
  )
  
}

nrmse(solubility_test, truth = solubility, estimate = prediction)
