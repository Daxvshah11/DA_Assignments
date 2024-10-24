# all imports here
from sklearn.svm import SVC
import sklearn as svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


# importing dataset from csv file
og_train_data = pd.read_csv("Customer_train.csv")
og_test_data = pd.read_csv("Customer_test.csv")


# printing number of rows in the datasets
print("Number of rows in the training dataset: ", len(og_train_data))
print("Number of rows in the test dataset: ", len(og_test_data))


# function to check skewness & fill missing values with mean or median based on skewness
def fill_missing_based_on_skewness(df, column):
    if df[column].isnull().sum() > 0:
        skewness = df[column].skew()
        if abs(skewness) > 0.5:
            fill_value = df[column].median()
            method = "median"
        else:
            fill_value = df[column].mean()
            method = "mean"

        df[column].fillna(fill_value, inplace=True)
        print(f"Filled missing '{column}' with {method}: {fill_value}\n")


# function to check skewness & transform data based on it using log transformation
def transform_if_skewed(df, column):
    if (
        df[column].isnull().sum() == 0
    ):  # ensuring no missing values before transformation
        skewness = df[column].skew()
        print(f"Skewness of '{column}': {skewness}")

        # applying log transformation here if skewness > 0.5
        if abs(skewness) > 0.5:
            df[column] = np.log1p(df[column])
            print(f"Applied log transformation to '{column}' due to skewness.\n")
        else:
            print(
                f"'{column}' is not significantly skewed; no transformation applied.\n"
            )
    else:
        print(
            f"Column '{column}' has missing values, handle them before transformation.\n"
        )


# function to fill missing values of categorical column with mode
def fill_missing_categorical(df, column):
    if df[column].isnull().sum() > 0:
        mode_value = df[column].mode()[0]
        df[column].fillna(mode_value, inplace=True)
        print(f"Filled missing '{column}' with mode: {mode_value}\n")


# making new df and processing in that one
processed_train_data = og_train_data.copy()

# dropping ID column since no use of it in training classifier
processed_train_data.drop("ID", axis=1, inplace=True)

# calling missing values function on columns as below
fill_missing_based_on_skewness(processed_train_data, "Work_Experience")
fill_missing_based_on_skewness(processed_train_data, "Family_Size")

# calling missing values function on categorical columns
fill_missing_categorical(processed_train_data, "Var_1")
fill_missing_categorical(processed_train_data, "Gender")
fill_missing_categorical(processed_train_data, "Ever_Married")
fill_missing_categorical(processed_train_data, "Graduated")
fill_missing_categorical(processed_train_data, "Profession")
fill_missing_categorical(processed_train_data, "Spending_Score")

# calling transformation function on columns as below
transform_if_skewed(processed_train_data, "Work_Experience")
transform_if_skewed(processed_train_data, "Family_Size")
transform_if_skewed(processed_train_data, "Age")


# initializing the MinMaxScaler
scaler = MinMaxScaler()

# selecting columns to be scaled
columns_to_scale = ["Age", "Work_Experience", "Family_Size"]

# scaling columns to the range [0, 1]
processed_train_data[columns_to_scale] = scaler.fit_transform(
    processed_train_data[columns_to_scale]
)


# list of categorical columns to be encoded
categorical_columns = [
    "Gender",
    "Ever_Married",
    "Graduated",
    "Profession",
    "Spending_Score",
    "Var_1",
]

# label encoding for binary categorical features
label_encoder = LabelEncoder()
for col in ["Gender", "Ever_Married", "Graduated"]:
    processed_train_data[col] = label_encoder.fit_transform(processed_train_data[col])

# one-hot encoding for multi-class categorical features
processed_train_data = pd.get_dummies(
    processed_train_data,
    columns=["Profession", "Spending_Score", "Var_1"],
    drop_first=True,
)


# # printing
# processed_train_data


# preparing testing dataset
processed_test_data = og_test_data.copy()

# dropping ID column since even training data has total only 22 columns (ID was not there)
processed_test_data.drop("ID", axis=1, inplace=True)

# same preprocessing steps to the test data
fill_missing_based_on_skewness(processed_test_data, "Work_Experience")
fill_missing_based_on_skewness(processed_test_data, "Family_Size")

fill_missing_categorical(processed_test_data, "Var_1")
fill_missing_categorical(processed_test_data, "Gender")
fill_missing_categorical(processed_test_data, "Ever_Married")
fill_missing_categorical(processed_test_data, "Graduated")
fill_missing_categorical(processed_test_data, "Profession")
fill_missing_categorical(processed_test_data, "Spending_Score")

transform_if_skewed(processed_test_data, "Work_Experience")
transform_if_skewed(processed_test_data, "Family_Size")
transform_if_skewed(processed_test_data, "Age")

# scaling test dataset with same scaler
columns_to_scale = ["Age", "Work_Experience", "Family_Size"]
processed_test_data[columns_to_scale] = scaler.transform(
    processed_test_data[columns_to_scale]
)

# label encoding for binary categorical features
label_encoder = LabelEncoder()
for col in ["Gender", "Ever_Married", "Graduated"]:
    processed_test_data[col] = label_encoder.fit_transform(processed_test_data[col])

# one-hot encoding for multi-class categorical features
processed_test_data = pd.get_dummies(
    processed_test_data,
    columns=["Profession", "Spending_Score", "Var_1"],
    drop_first=True,
)

# # printing
# processed_test_data


# function to train an SVM for a given class pair
def train_ovo_classifiers(X_train, y_train, class_labels):
    # dictionary to store classifiers for each pair
    classifiers = {}

    # generating all possible class pairs
    pairs = list(itertools.combinations(class_labels, 2))

    # iterating over all pairs
    for class1, class2 in pairs:
        # selecting only data points that belong to the two classes
        idx = np.where((y_train == class1) | (y_train == class2))
        X_pair = X_train[idx]
        y_pair = y_train[idx]

        # converting y_train into binary labels (class1 = 1 & class2 = -1)
        y_pair = np.where(y_pair == class1, 1, -1)

        # training an SVM classifier on the reduced dataset
        clf = SVC(kernel="rbf", C=10, gamma=0.2)
        clf.fit(X_pair, y_pair)

        # storing classifier for this pair
        classifiers[(class1, class2)] = clf

    return classifiers


# function to predict class for a new data point, using all classifiers
def ovo_predict(X_test, classifiers):
    # all votes list
    votes = []

    # iterating over data points in the test set
    for x in X_test:
        # voting dictionary for each data point
        class_votes = defaultdict(int)

        # iterating through all classifiers & getting predictions
        for (class1, class2), clf in classifiers.items():
            pred = clf.predict([x])

            # assigning votes based on the prediction (either class1 or class2)
            if pred == 1:
                class_votes[class1] += 1
            else:
                class_votes[class2] += 1

        # getting the class with max votes
        final_class = max(class_votes, key=class_votes.get)
        votes.append(final_class)

    return np.array(votes)


# getting X_test
X_test = processed_test_data.values

# initialising class labels with unique values in Segmentation column
class_labels = processed_train_data["Segmentation"].unique()

# creating a copy of the processes df & making X_train (np array)
X_train_copy = processed_train_data.copy()
X_train_copy.drop("Segmentation", axis=1, inplace=True)
X_train = X_train_copy.values

# similarly getting y_train
y_train = processed_train_data["Segmentation"].values

# calling function to train the OvO classifiers
classifiers = train_ovo_classifiers(X_train, y_train, class_labels)

# making predictions for test data points
predictions = ovo_predict(X_test, classifiers)

# converting the predictions into a pandas DataFrame
predictions_df = pd.DataFrame(predictions, columns=["predictions"])

# saving the df to a CSV file
predictions_df.to_csv("ovo.csv", index=False)


# splitting the data into features (X) and labels (y)
y = processed_train_data["Segmentation"].values
new_copy = processed_train_data.copy()
X = new_copy.drop("Segmentation", axis=1).values

# splitting into 80% training and 20% validation set (randomly)
new_train_set, new_valid_set, y_train_new, y_valid_new = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create a dictionary to simulate 'train_ova_classifiers' and 'ova_predict' for OvA case
new_classifiers = train_ovo_classifiers(new_train_set, y_train_new, class_labels)

# making predictions on the validation set
predictions = ovo_predict(new_valid_set, classifiers)


# making a confusion matrix for the above split
def plot_confusion_matrix(cm, classes, title="Confusion matrix", cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# confusion matrix
cm = confusion_matrix(y_valid_new, predictions)
plt.figure(figsize=(8, 8))
plot_confusion_matrix(cm, class_labels, title="Confusion Matrix for OvO")
plt.savefig("ovo_confusion_matrix.png")


# classification report image
report = classification_report(
    y_valid_new, predictions, target_names=class_labels, output_dict=True
)
df = pd.DataFrame(report).transpose()
plt.figure(figsize=(12, 6))
sns.heatmap(df, annot=True, cmap="coolwarm")
plt.title("Classification Report for OvO")
plt.savefig("ovo_classification_report.png")

# accuracy score
accuracy = accuracy_score(y_valid_new, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
