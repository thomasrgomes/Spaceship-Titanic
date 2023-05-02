## spaceship titanic - kaggle project ##

## load libraries 
library(tidyverse)
library(corrplot)
library(GGally)
library(tidymodels)
library(bonsai)
library(readr)

## load data 
train <- read_csv('data/train.csv', show_col_types = FALSE)
test <- read_csv('data/test.csv')

## Exploratory Data Analysis
head(train)


# visualize corralations
train %>%
  select_if(is.numeric) %>%
  cor(use = "complete.obs", method = "spearman") %>%
  corrplot(order = 'AOE')

train %>%
  filter(!CryoSleep) %>%
  select(Transported, RoomService:VRDeck) %>%
  mutate_if(is.numeric, ~ log(. + 1)) %>%
  filter(complete.cases(.)) %>%
  filter_if(is.numeric, ~ . > 0) %>%
  ggpairs(aes(col = Transported, alpha = 0.7), columns = 2:6,
          lower = list(continuous = wrap("points", alpha = 0.8, size = 0.4)))

# pre-processing

pre_process <- function(dat){
  dat %>%
    separate(Cabin, c("CabinDeck", "CabinNum", "CabinSide"), sep = "\\/", remove = FALSE) %>%
    separate(PassengerId, c("PassengerGroup", "PassengerNum"), sep = "_", remove = FALSE) %>%
    add_count(PassengerGroup, name = "GroupSize") %>%
    mutate(
      SpendLuxury = RoomService + Spa + VRDeck,
      SpendRegular = FoodCourt + ShoppingMall,
      FoodPreference = RoomService / (FoodCourt + RoomService),
      ActivityPreference = VRDeck / (Spa + VRDeck),
      PassengerGroup = as.numeric(PassengerGroup),
      FamilyName = as.factor(str_extract(Name, "[A-z]+$"))
    ) %>%
    group_by(PassengerGroup) %>%
    mutate(
      GroupVIP = any(VIP, na.rm = TRUE),
    ) %>%
    add_count(FamilyName, name = "FamilySize") %>%
    ungroup() %>%
    mutate_if(is.character, as.factor) %>%
    select(-Name, -Cabin, -Name, -PassengerNum, -CabinNum, -FamilyName)
}

# univariate associations
## using logistic regression to see if any variables have a relationship with their given outcome
train %>%
  pre_process() %>% 
  select(-PassengerId) %>%
  mutate_at(vars(RoomService:VRDeck), ~ log(. + 1)) %>%
  mutate_at(vars(SpendLuxury:SpendRegular), ~ log(. + 1)) %>%
  glm(Transported ~ ., family = "binomial", data = .) %>%
  summary()

# training the data
## imputation using KNN

x.train <- 
  train %>%
  pre_process() %>%
  select(-PassengerId) %>%
  mutate(Transported = as.factor(Transported))



## using the recipe function to impute missing values, 
## convert categorical variables (besides the outcome) to dummies
rec <-
  # using tidymodels to set Transported as outcome variable and all others as 
  # predictor variables
  recipe(Transported ~ ., data = x.train) %>%
  ## use knn to impute missing values for all predictors
  step_impute_knn(all_predictors()) %>%
  ## create dummy variables for all categorical (nominal) variables, except for
  ## outcome variable (Transported)
  step_dummy(all_nominal(), -all_outcomes())

## create boosted tree model framework, with tuning of parameters already set
model <-
  # initialize boosted tree model
  boost_tree(
    # set params - optimization
    trees = tune(),
    tree_depth = tune(),
    min_n = tune(),
    mtry = tune(),
    learn_rate = tune()
  ) %>%
  # sets engine of model to LightGBM - a model boosting framework
  set_engine("lightgbm") %>%
  # set the mode of the model to classification, as we are using a binary outcome
  # variable
  set_mode("classification")

## set the hyper-parameters to be used in the model
grid <- 
  ## This function generates a Latin hypercube grid, which is a sampling design
  ## that ensures coverage of the parameter space with evenly distributed values.
  grid_latin_hypercube(
    # set parameters for model values from above 
    trees(c(500, 1500)),
    tree_depth(c(5, 15)),
    min_n(c(2, 20)),
    finalize(mtry(), x.train),
    learn_rate(),
    # number of combinations of above parameters to be used
    size = 100
  )

## use the workflows package to execute model and parameters at once
res <- 
  workflow() %>%
  ## add in recipe 
  add_recipe(rec) %>%
  add_model(model) %>%
  ## use 10 fold cross validation for optimzation and tuning of model
  tune_grid(
    resamples = vfold_cv(x.train, 10),
    grid = grid,
    metrics = metric_set(accuracy)
  )

## Examining model fits
res %>%
  collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  select(mean, trees:learn_rate) %>%
  mutate(learn_rate = log(learn_rate)) %>%
  filter(mean > 0.7) %>%
  pivot_longer(trees:learn_rate,
               values_to = "value",
               names_to = "parameter") %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_grid(~parameter, scales = "free_x") +
  labs(x = NULL, y = "Accuracy")

show_best(res, metric = "accuracy")

## Finalize model 
final <- workflow() %>%
  add_recipe(rec) %>%
  add_model(model) %>%
  finalize_workflow(select_best(res, metric = "accuracy")) %>%
  fit(data = train)

final$fit$fit$fit %>%
  lgb.importance() %>% 
  as_tibble() %>% 
  arrange(Gain) %>% 
  ggplot(aes(fct_reorder(Feature, Gain), Gain)) + geom_col() + coord_flip()

test <- read_csv("../input/spaceship-titanic/test.csv", show_col_types = FALSE) %>%
  preprocess()

select(test, PassengerId) %>%
  bind_cols(predict(final, test)) %>%
  rename("Transported" = ".pred_class") %>%
  mutate(Transported = str_to_title(as.character(Transported))) %>%
  write_csv("submission.csv")

