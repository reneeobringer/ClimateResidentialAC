# Project: Climate and Residential Air Conditioning (Pecan Street Data)
# Code By: Simon Pezalla and Renee Obringer
# Last Updated: 18 October 2022

# ORGANIZATION:
# This code is organized in sections, each denoted by multiple #
# LOAD AND CLEAN DATA -- Load data and pre-process as necessary
# TUNING + FITTING MODELS -- Initial model tuning, training, and testing
# BEST MODEL - BART ONLY -- Using the best model, rerun outside the tidymodels framework for easier interpretation
# ANALYSIS + FIGURES -- Analyze results and visualization

# libraries
library(tidymodels) 
library(ggplot2)
library(reshape2)

# set path 
# NOTE: set this path to the folder on your personal machine which contains the downloaded data 
# for example: path <- '/Users/rqo5125/Downloads/ClimateResidentialAC'
path <- ''

# OPTIONAL: Create a directory for output files
outputdir <- paste(path,'/output/', sep = '') 
dir.create(outputdir)

################### LOAD AND CLEAN DATA ###################

setwd(paste(path, '/inputdata', sep = ''))
# load data
austin <- read.csv('austindata.csv')
newyork <- read.csv('newyorkdata.csv')
sandiego <- read.csv('sandiegodata.csv')

# remove columns
austin <- austin[,-c(1,4,6)]
newyork <- newyork[,-c(1,4,6)]
sandiego <- sandiego[,-c(1,4,6)]

# get unique ids
id_var_austin <- unique(austin['dataid'])
id_var_ny <- unique(newyork['dataid'])
id_var_sd <- unique(sandiego['dataid'])

# Removes all houses with median AC use less than 2 kWh

cities <- list(austin, newyork, sandiego)

id_variables <- list(id_var_austin,id_var_ny,id_var_sd)
ids_with_no_ac <- list(c(), c(), c())

for (c in 1:3) {
  for (i in 1:nrow(id_variables[[c]])) {
    if (median(cities[[c]][which(cities[[c]]$dataid == id_variables[[c]][i,]),]$airconditioning) < 2) {
      ids_with_no_ac[[c]][i] <- id_variables[[c]][i,]
    }
  }
}

austin <- austin[!(austin$dataid %in% ids_with_no_ac[[1]]),]
newyork <- newyork[!(newyork$dataid %in% ids_with_no_ac[[2]]),]
sandiego <- sandiego[!(sandiego$dataid %in% ids_with_no_ac[[3]]),]

# remove zeroes + negatives
austin[austin$airconditioning <= 0, 3] <- NA; austin <- na.omit(austin)
newyork[newyork$airconditioning <= 0, 3] <- NA; newyork <- na.omit(newyork)
sandiego[sandiego$airconditioning <= 0, 3] <- NA; sandiego <- na.omit(sandiego)

cities <- list(austin, newyork, sandiego)

setwd(outputdir)
save.image('cleaneddata.rdata')

################### TUNING + FITTING MODELS ##############

setwd(outoutdir)
load('cleaneddata.rdata')

# get unique ids
id_vars <- list(unique(cities[[1]]['dataid']), unique(cities[[2]]['dataid']), unique(cities[[3]]['dataid']))

# initialize storage variables
metrics <- list(list(),list(),list())

# initialize models
glm_mod <- linear_reg(mode = 'regression', penalty = 0) %>% set_engine('glmnet', family = 'gaussian')                                         # GLM
gam_mod <- gen_additive_mod(mode = 'regression', select_features = T, adjust_deg_free = tune()) %>% set_engine("mgcv")                        # GAM
mars_mod <- mars(num_terms = tune(), prod_degree = tune(), prune_method = tune()) %>% set_engine("earth") %>% set_mode("regression")          # MARS
cart_mod <- decision_tree(tree_depth = tune(), cost_complexity = tune(), min_n = tune()) %>% set_engine("rpart") %>% set_mode("regression")   # CART
rf_mod <- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>% set_engine("randomForest") %>% set_mode("regression")                # RANDOM FOREST
bart_mod <- bart(trees = tune()) %>% set_engine("dbarts") %>% set_mode("regression")                                                          # BART
nnet_mod <- mlp(hidden_units = tune(), penalty = tune()) %>% set_engine("nnet") %>% set_mode("regression")                                    # NEURAL NETWORK

# loop through each city
for (c in 1:3) {
  # select city
  citydata <- cities[[c]]
  
  # loop through each house
  for (h in 1:nrow(id_vars[[c]])) {
    # select household
    hhdata <- cities[[c]][which(cities[[c]]$dataid == id_vars[[c]][1,]),]
    
    # set up cross validation
    cv <- vfold_cv(hhdata, v = 5)
    
    # create recipe
    model_recipe <- recipe(airconditioning ~ rh + Tc + Tw + Td + sWBGT + DI + HI + HUMIDEX, data = hhdata)
    
    # create workflows
    glm_wf <- workflow() %>% add_recipe(model_recipe) %>% add_model(glm_mod)
    gam_wf <- workflow() %>% add_recipe(model_recipe) %>% add_model(gam_mod, formula = airconditioning ~ rh + Tc + Tw + Td + sWBGT + DI + HI + HUMIDEX)
    mars_wf <- workflow() %>% add_recipe(model_recipe) %>% add_model(mars_mod)
    cart_wf <- workflow() %>% add_recipe(model_recipe) %>% add_model(cart_mod)
    rf_wf <- workflow() %>% add_recipe(model_recipe) %>% add_model(rf_mod)
    bart_wf <- workflow() %>% add_recipe(model_recipe) %>% add_model(bart_mod)
    nnet_wf <- workflow() %>% add_recipe(model_recipe) %>% add_model(nnet_mod)
    
    # OPTIONAL: tune models
    gam_grid <- grid_regular(adjust_deg_free(c(1,6)), levels = 6)
    gam_tune_results <- gam_wf %>% tune_grid(resamples = cv, grid = gam_grid)
    gam_best_params <- gam_tune_results %>% select_best(metric = "rmse")
    gam_wf2 <- gam_wf %>% finalize_workflow(gam_best_params)
    
    cart_grid <- grid_regular(tree_depth(), cost_complexity(), min_n(), levels = 3)
    cart_tune_results <- cart_wf %>% tune_grid(resamples = cv, grid = cart_grid)
    cart_best_params <- cart_tune_results %>% select_best(metric = "rmse")
    cart_wf2 <- cart_wf %>% finalize_workflow(cart_best_params)
    
    mars_grid <- grid_regular(num_terms(c(4,16)), prod_degree(c(1,3)), prune_method(c('backward', 'forward', 'none')), levels = 3)
    mars_tune_results <-  mars_wf %>% tune_grid(resamples = cv, grid = mars_grid)
    mars_best_params <- mars_tune_results %>% select_best(metric = "rmse")
    mars_wf2 <- mars_wf %>% finalize_workflow(mars_best_params)
    
    rf_grid <- grid_regular(mtry(c(2,6)), trees(c(100,750)), min_n(c(5, 15)), levels = 3)
    rf_tune_results <- rf_wf %>% tune_grid(resamples = cv, grid = rf_grid)
    rf_best_params <- rf_tune_results %>% select_best(metric = "rmse")
    rf_wf2 <- rf_wf %>% finalize_workflow(rf_best_params)
    
    bart_grid <- grid_regular(trees(c(100, 750)), levels = 3)
    bart_tune_results <- bart_wf %>% tune_grid(resamples = cv, grid = bart_grid)
    bart_best_params <- bart_tune_results %>% select_best(metric = "rmse")
    bart_wf2 <- bart_wf %>% finalize_workflow(bart_best_params)
    
    nnet_grid <- grid_regular(hidden_units(), penalty(), levels = 3)
    nnet_tune_results <- nnet_wf %>% tune_grid(resamples = cv, grid = nnet_grid)
    nnet_best_params <- nnet_tune_results %>% select_best(metric = "rmse")
    nnet_wf2 <- nnet_wf %>% finalize_workflow(nnet_best_params)
    
    # fit models
    glm_fit <- glm_wf %>% fit_resamples(cv)
    gam_fit <- gam_wf2 %>% fit_resamples(cv)
    mars_fit <- mars_wf2 %>% fit_resamples(cv)
    cart_fit <- cart_wf2 %>% fit_resamples(cv)
    rf_fit <- rf_wf2 %>% fit_resamples(cv)
    bart_fit <- bart_wf2 %>% fit_resamples(cv)
    nnet_fit <- nnet_wf2 %>% fit_resamples(cv)
    
    # store results
    metrics[[c]][['glm']][[h]] <- collect_metrics(glm_fit)
    metrics[[c]][['gam']][[h]] <- collect_metrics(gam_fit)
    metrics[[c]][['mars']][[h]] <- collect_metrics(mars_fit)
    metrics[[c]][['cart']][[h]] <- collect_metrics(cart_fit)
    metrics[[c]][['rf']][[h]] <- collect_metrics(rf_fit)
    metrics[[c]][['bart']][[h]] <- collect_metrics(bart_fit)
    metrics[[c]][['nnet']][[h]] <- collect_metrics(nnet_fit)
  }
}

setwd(outputdir)
save.image('modelrundata.rdata')

################### BEST MODEL - BART ONLY #################

library(dbarts)

setwd(outputdir)
load('cleaneddata.rdata')

# get unique ids
id_vars <- list(unique(cities[[1]]['dataid']), unique(cities[[2]]['dataid']), unique(cities[[3]]['dataid']))

# initialize fit storage variable
bart_fit_cities <- list()

# loop through each city
for (c in 1:3) {
  # select city
  citydata <- cities[[c]]
  bart_mod_fin <- list()
  
  # loop through each house
  for (h in 1:nrow(id_vars[[c]])) {
    # select household
    hhdata <- cities[[c]][which(cities[[c]]$dataid == id_vars[[c]][1,]),]
    
    # sample data
    sample_rows <- sample(nrow(hhdata), size = 0.8*nrow(hhdata), replace = F) # sample 80% of the data WITHOUT replacement
    
    hhdata_train <- hhdata[sample_rows,] # create the training dataset (80%)
    hhdata_test <- hhdata[-sample_rows,] # create the test dataset (20%)
    
    # fit model 
    bart_mod_fin[[h]] <- bart(hhdata_train[,4:11], hhdata_train[,3], x.test = hhdata_test[,4:11], ntree = 425, keeptrees = TRUE)
  }
  bart_fit_cities[[c]] <- bart_mod_fin
}

setwd(outputdir)
save.image('bartmodelrun.rdata')

################### ANALYSIS + FIGURES ##############

setwd(outputdir)

load('modelrundata.rdata')
load('bartmodelrun.rdata')

# re-organize metrics list

# set city names
cityname <- c('Austin', 'New York', 'San Diego')

# initialize storage variable
allresults <- c()

for (j in 1:3) {
  # extract city metrics
  cityresults <- metrics[[j]]
  
  # get length variable
  n <- length(cityresults[['glm']])
  
  # initialize temporary storage variable
  citytmp <- c()
  
  # loop through all households
  for (i in 1:n) {
    citytmp <- rbind(citytmp, sapply(cityresults, function(x) {x[[i]][[3]]}))
  }
  
  # create label data
  households <- rep(1:n, each = 2)
  measure <- rep(c('rmse','rsq'), n)
  city <- rep(cityname[j],n*2)
  
  # combine data for a specific city
  citytmp <- cbind(citytmp, households, measure, city)
  
  # combine data from all cities
  allresults <- rbind(allresults, citytmp)
}

allresults <- data.frame(allresults)

allresults_long <- melt(allresults, id.vars = c('households','city', 'measure'), measure.vars = c('glm','gam','mars','cart','rf','bart','nnet'))
allresults_long$value <- as.numeric(allresults_long$value) 

# get mean values
averages <- aggregate(allresults_long$value, list(allresults_long$city, allresults_long$measure, allresults_long$variable), function(x) mean(x, na.rm = T))

# t-test for MARS v BART
atxMARS <- allresults_long[which(allresults_long$city == 'Austin' & allresults_long$measure == 'rmse' & allresults_long$variable == 'mars'),5]
atxBART <- allresults_long[which(allresults_long$city == 'Austin' & allresults_long$measure == 'rmse' & allresults_long$variable == 'bart'),5]
t.test(atxMARS, atxBART)

nyMARS <- allresults_long[which(allresults_long$city == 'New York' & allresults_long$measure == 'rmse' & allresults_long$variable == 'mars'),5]
nyBART <- allresults_long[which(allresults_long$city == 'New York' & allresults_long$measure == 'rmse' & allresults_long$variable == 'bart'),5]
t.test(nyMARS, nyBART)

sdMARS <- allresults_long[which(allresults_long$city == 'San Diego' & allresults_long$measure == 'rmse' & allresults_long$variable == 'mars'),5]
sdBART <- allresults_long[which(allresults_long$city == 'San Diego' & allresults_long$measure == 'rmse' & allresults_long$variable == 'bart'),5]
t.test(sdMARS, sdBART)

# plots

bartresults <- allresults_long[which(allresults_long$measure == 'rmse'),]
means <- aggregate(bartresults$value, list(bartresults$city, bartresults$variable), mean)
mins <- aggregate(bartresults$value, list(bartresults$city, bartresults$variable), min)
maxs <- aggregate(bartresults$value, list(bartresults$city, bartresults$variable), max)

plotdata <- cbind(means, mins$x, maxs$x)
names(plotdata) <- c('City', 'Model', 'Mean', 'Minimum', 'Maximum')

setwd(outputdir)
pdf('rmse_variation.pdf', width = 11, height = 7)
ggplot(plotdata, aes(x = City, y = Mean, fill = Model)) + geom_bar(stat = 'identity', position = position_dodge(), color = '#d3d3d3') +
  geom_errorbar(aes(x = City, ymin = Minimum, ymax = Maximum), position = position_dodge()) +
  scale_fill_manual(values = c('#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69'), 
                    labels = c('GLM','GAM', 'MARS','CART','RF','BART','NN')) + ylab('RMSE (kWh)') +
  theme_light() + theme(text = element_text(size = 16))
dev.off() 
  
# Variable Importance

# source code for variable importance from https://github.com/cjcarlson/embarcadero
source(paste(outputdir,'/varimp.R',sep = ''))

# initialize storage variable
VIbart_all <- list()

# loop through cities
for (c in 1:3) {
  # initialize storage variable
  VIbart <- list()
  # loop through households
  for (h in 1:length(bart_fit_cities[[c]])) {
    # find and store variable importance
    VIbart[[h]] <- varimp(bart_fit_cities[[c]][[h]])
  }
  # store variable importance
  VIbart_all[[c]] <- VIbart
}

# reformat VI data

allVI <- c()

for (c in 1:3) {
  # extract city metrics
  cityVI <- VIbart_all[[c]]
  
  # get length variable
  n <- length(cityVI)
  
  # initialize temporary storage variable
  citytmp <- c()
  
  # loop through all households
  for (i in 1:n) {
    citytmp <- rbind(citytmp, cityVI[[i]])
  }
  
  # create label data
  households <- rep(1:n, each = 8)
  city <- rep(cityname[c],8)
  
  # combine data for a specific city
  citytmp <- cbind(citytmp, households, city)
  
  # combine data from all cities
  allVI <- rbind(allVI, citytmp)
}
 
# get means, mins, maxs

means <- aggregate(allVI$varimps, list(allVI$city, allVI$names), mean)
mins <- aggregate(allVI$varimps, list(allVI$city, allVI$names), min)
maxs <- aggregate(allVI$varimps, list(allVI$city, allVI$names), max)

# plot variable importance

plotdata <- data.frame(means, mins$x, maxs$x)
names(plotdata) <- c('City', 'Predictor', 'Mean', 'Minimum', 'Maximum')

setwd(outputdir)
pdf('VIall_vertical.pdf', width = 6, height = 6.5)
ggplot(plotdata) + geom_point(aes(y = Predictor, x = Mean, color = City), size = 2) +
  geom_errorbar(aes(xmin = Minimum, xmax = Maximum, y = Predictor, color = City), width = 0.2) +
  theme_light() + theme(text = element_text(size = 16), legend.position="bottom") +
  scale_color_manual(values = c('#66c2a5','#fc8d62','#8da0cb')) + xlab('Variable Importance') +
  scale_y_discrete(limits=rev)
dev.off()  
  
# Partial Dependence  
  
setwd(outputdir)
load('cleaneddata.rdata')

# get unique ids
id_vars <- list(unique(cities[[1]]['dataid']), unique(cities[[2]]['dataid']), unique(cities[[3]]['dataid']))

# initialize fit storage variable
pdbart_fit_cities <- list()

# loop through each city
for (c in 1:3) {
  # select city
  citydata <- cities[[c]]
  pdbart_mod_fin <- list()
  
  # loop through each house
  for (h in 1:nrow(id_vars[[c]])) {
   # select household
   hhdata <- cities[[c]][which(cities[[c]]$dataid == id_vars[[c]][1,]),]

   # sample data
   sample_rows <- sample(nrow(hhdata), size = 0.8*nrow(hhdata), replace = F) # sample 80% of the data WITHOUT replacement

   hhdata_train <- hhdata[sample_rows,] # create the training dataset (80%)
   hhdata_test <- hhdata[-sample_rows,] # create the test dataset (20%)

   # fit model
   pdbart_mod_fin[[h]] <- pdbart(hhdata_train[,4:11], hhdata_train[,3], x.ind = c(1,2), ntree = 750, keeptrees = TRUE)
  }
  
  pdbart_fit_cities[[c]] <- pdbart_mod_fin
}

setwd(outputdir)
# plot all households

lims <- matrix(c(0, 0, 0, 80, 40, 50), ncol = 2)
for (c in 1:3) {
  for (h in 1:length(pdbart_fit_cities[[c]])) {
    pdf(paste('pdp_city_', c, '_hh_', h, '.pdf', sep = ''), width = 8, height = 4)
    par(mfrow=c(1,2))
    plot(pdbart_fit_cities[[c]][[h]], xind = c(1,4),  cols = c('black', 'red'), ylim = c(lims[c,1], lims[c,2]))
    dev.off()
  }
}

