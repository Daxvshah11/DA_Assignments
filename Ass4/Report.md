# Team 8 : Assingment 4 (Report)

## Assumptions

* For Task 1, when training the SVM classifiers, `kernel : rbf` is used because it was noticed that there were better results in case of __rbf__ as compared to __linear__




## Task 1

### Choosing Tuning Parameters

* We have manually run multiple runs on different combinations of `C & Gamma` with splitting of the training set

* We used `validation sets` to calculate accuracy for the purpose of tuning both the above parameters

* Some of the best balanced combinations we found are as below (C | Gamma | Accuracy):
    * > 10 | 0.1 | 0.3967391304347826
    * > 20 | 0.1 | 0.4076086956521739
    * > 10 | 0.2 | 0.44021739130434784
    * > 20 | 0.2 | 0.4483695652173913

* Currently, we have kept the combo of `C=10 & Gamma=0.2` as the most balanced one to avoid any kind of __underfit or overfit__

* Although, these values can be easily changed according to the purpose in the functions



## Task 2




## Task 3