# Team 8 : Assingment 4 (Report)

## Assumptions

* For Task 1, when training the SVM classifiers, `kernel : rbf` is used because it was noticed that there were better results in case of __rbf__ as compared to __linear__




## Task 1

### Choosing Tuning Parameters

* We have run a GridSearch function (trying all possible combinations) on the dataset

* It uses Cross Validation on the dataset to see which combination of the tuning parameters, namely `C & Gamma` for __rbf__ kernel are the best suited, finding a balance between underfit and overfit

* We are relying on its results to believe that the model trained with those tuning parameters would give us the `relatively best accuracy` model & predictions

* It has been run once in OvO file and those same values are directly used in the OvA file too since the dataset remains same



## Task 2




## Task 3