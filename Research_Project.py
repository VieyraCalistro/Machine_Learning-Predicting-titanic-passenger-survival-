import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import StandardScaler



def Research():

    # Reading in the csv file
    df = pd.read_csv("titanic_passengers.csv", encoding="latin1")
    print(df)
    os.system("pause")

    # Doing a quick health check on the data set
    print(f'\n{df.info()}\n')
    os.system("pause")

    # Checking a class imbalance
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Survived")
    plt.savefig("Class_Balance.png")
    print(f'Class_Balance.png saved!\n')
    plt.show()
    # We have a class imbalance of about 200 difference.

    # Checking distribution of Survived
    fig, ax = plt.subplots()
    df["Survived"].plot.hist()
    plt.savefig("Survived_histogram.png")
    print(f'\nSurvived_histogram.png saved!\n')
    plt.show()


    # Perform Exploratory Data Analysis
    # y = f(x)
    # y = Survived
    # X = All other inputs (that will be useful)


    # Here I am going to replace missing data in the "Age" feature to help performance of this model
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    X = imp.fit_transform(df[["Age"]])
    df["Age"] = X
    print(f'\n{df.info()}\n')
    
    # Here I am taking out the "Cabin" feature because we have way too much info that is missing
    # I don't want to just fill the rest up random values.
    # I am also taking out "Name" and "PassengerId" because these are high in cardinality and they will
    # not be much help to me here for now. 
    df = df[["Survived", "Pclass", "Sex", "Age",
             "SibSp", "Parch", "Ticket", "Fare", "Embarked"]]
    print(f'\n{df}\n')

    # Here I am filling the missing values in the "Embarked" feature to help performance
    imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    X = imp.fit_transform(df[["Embarked"]])
    df["Embarked"] = X

    print(f'\n{df.info()}\n')
    os.system("pause")


    # Here I am looking for signs of relationships with data, specifically with the survived column
    sns.pairplot(df)
    plt.subplots_adjust(bottom=0.1)
    plt.savefig("DataFrame_pairplot.png")
    print(f'DataFrame_pairplot.png saved!\n')
    plt.show()
    # I feel like the Survived plots are being overlapped so the comparison becomes useless.
    # No relationship noticed at all with Survived and the available plottable columns of data using
    # this pairplot.
    # Lets take a closer look with a corelation matrix on a heat map.
    

    # Here I am checking for correlation with Survived
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, ax=ax)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig("Heatmap1.png")
    print(f'Heatmap1.png saved!\n')
    plt.show()

    # Interesting but it kind of makes sense, I noticed that depending on the type of ticket class
    # the people obtain, correlated with the survival column. I wonder if the higher class had a slightly
    # higher survival rate, and I say that because the price of the fare also played a role in the survival column.
    # This doesn't necessarily mean that higher class tickets holders had a slightly higher survival rate,
    # it just means that it correlated slightly with whether they lived or not.


    # Now my plan is to binarize some data so I can make it plottable
    binarize = LabelBinarizer()
    vals = np.array(df["Sex"])
    dummy_codes = binarize.fit_transform(vals)
    print(f'\n\{df["Sex"]}\n{dummy_codes[:][:] }\n')
    
    # Gathering dummy_codes into one column
    pca = PCA(n_components=1, random_state=100)
    dummy_codes_compressed = pca.fit_transform(dummy_codes)
    df["Sex"] = dummy_codes_compressed
    print(f'\n{df["Sex"]}\n')


    # # Here I tried to binarize "Ticket" feature but Came out with a lot of zero's
    # # I'm concluding that this feature is also high in cardinality and going to take it off of the my data for now.

    #vals = np.array(df["Ticket"])
    #dummy_codes = binarize.fit_transform(vals)
    #df["Ticket"] = dummy_codes
    #print(dummy_codes)

    # Verified, we have 681 unique samples for this Ticket feature
    unique = np.unique(df["Ticket"])
    print(f'{unique}')
    print(f'\nUnique samples in "Ticket" feature: {sum(1 for each in unique)}\n')


    # Took the Ticket feature out as to it will not be much help to us.
    df = df[["Survived", "Pclass", "Sex", "Age",
             "SibSp", "Parch", "Fare", "Embarked"]]
    print(f'\n{df.info()}\n')


    # Categorical encoding Embarked feature so we can evaluate it as well.
    vals = np.array(df["Embarked"])
    dummy_codes = binarize.fit_transform(vals)
    print(f'\n\{df["Embarked"]}\n{dummy_codes[:][:] }\n')

    # Gathering dummy_codes into one column
    pca = PCA(n_components=1, random_state=100)
    dummy_codes_compressed = pca.fit_transform(dummy_codes)
    df["Embarked"] = dummy_codes_compressed
    print(f'\n{df["Embarked"]}\n')
    

    # Let's do another matrix correlation using a heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, ax=ax)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig("Heatmap2.png")
    print(f'Heatmap2.png saved!\n')
    plt.show()
    # Wow the "Sex" feature is showing the strongest correlation towards the "Survived" feature.


    # Time to split the Data
    yOutput = np.array(df["Survived"])
    xInput = df[["Pclass", "Sex", "Age",
             "SibSp", "Parch", "Fare", "Embarked"]].values


    X_train, X_test, y_train, y_test = train_test_split(xInput, yOutput, shuffle=False, random_state=100)


    # Here I am finding X_best and decided to keep the two best
    kselect = SelectKBest(k=2, score_func=f_classif)
    X_best = kselect.fit_transform(X_train, y_train)
    X_test = kselect.transform(X_test)
    print(f'\nX_best.shape: {X_best.shape}\n')
    print(f'kselect.scores_: {kselect.scores_}\n')
    print(f'kselect.get_support(): {kselect.get_support()}\n')

    fig, ax = plt.subplots()
    plt.scatter(X_best[:,0], X_best[:,1], c=y_train)
    plt.savefig("SelectKbest_Scatterplot.png")
    print(f'SelectKbest_Scatterplot.png saved!\n')
    plt.show()


    # Sometimes we do PCA if the scores are all about the same and we want to further reduce the deminsion of good scores
    # no need to perform more further compression with PCA since the major kselect scores were kept
    # just checking explained_variance_ratio_ and plotting a visual here to see the output.
    pca = PCA(n_components=2, random_state=100)
    X_best = pca.fit_transform(X_best)
    X_test = pca.transform(X_test)

    fig, ax = plt.subplots()
    plt.scatter(X_best[:,0], X_best[:,1], c=y_train)
    plt.savefig("Pca_Scatterplot.png")
    print(f'Pca_Scatterplot.png saved!\n')
    plt.show()
    print(f'\npca.explained_variance_ratio_: {pca.explained_variance_ratio_}\n')


    # Continuing on with new X_best data
    knn = KNeighborsClassifier()
    knn.fit(X_best, y_train)
    y_predict = knn.predict(X_test)
    print(f'\ny_predict:\n{y_predict}\n')

    # Going to print out test and train score
    print(f'\nKNN Train Score: {knn.score(X_best, y_train)}')
    print(f'KNN Test Score: {knn.score(X_test, y_test)}\n')
    print(f'r2: {metrics.r2_score(y_test, y_predict)}\n')

    # Printing out classification report
    print(f'\nClassification Report:\n{classification_report(y_test, y_predict)}\n')


    # Searching for outliers
    rgs = LocalOutlierFactor(n_neighbors=5)
    predictions = rgs.fit_predict(X_best)
    # inliers are 1, outliers are -1, remap as colors for output
    predictions = np.where(predictions == 1, 'blue', 'red')

    fig, ax = plt.subplots()
    plt.scatter(X_best[:,0], X_best[:, 1], c=predictions)
    plt.savefig("Outlier_scatterplot.png")
    print(f'Outlier_scatterplot.png saved!\n')
    plt.show()

    # Same issue here, I feel like they overlap so I'll print out the outlier index instead
    outliers = np.where(predictions == "red")
    print(f'\nOutlier indexes: {outliers[:][0]}\n')

    # Delete outlier(s) to see how much score improves
    X_Outliers_Removed = np.delete(X_best, 0, axis=0)
    X_Outliers_Removed = np.delete(X_Outliers_Removed, 0, axis=0)

    # Deleting one y_train sample to keep samples consistant
    y_train_Samples_Reduced = np.delete(y_train, 0)
    y_train_Samples_Reduced = np.delete(y_train_Samples_Reduced, 0)


    # Re-searching for outliers
    rgs = LocalOutlierFactor(n_neighbors=5)
    predictions = rgs.fit_predict(X_Outliers_Removed)
    # inliers are 1, outliers are -1, remap as colors for output
    predictions = np.where(predictions == 1, 'blue', 'red')

    # Verifying there are no more outliers within 5 neighbors
    outliers = np.where(predictions == "red")
    print(f'\nOutlier indexes: {outliers[:][0]}\n')

    # Re-checking scores to see if removing the outliers helped at all
    knn = KNeighborsClassifier()
    knn.fit(X_Outliers_Removed, y_train_Samples_Reduced)
    y_predict = knn.predict(X_test)

    # Going to print out test and train score
    print(f'\nKNN Train Score: {knn.score(X_Outliers_Removed, y_train_Samples_Reduced)}')
    print(f'KNN Test Score: {knn.score(X_test, y_test)}\n')
    print(f'r2: {metrics.r2_score(y_test, y_predict)}\n')
    print(f'\nClassification Report:\n{classification_report(y_test, y_predict)}\n')
    # Nice, showing improvement.


    # plotting confusion matrix to see the performance of the model
    fig, ax = plt.subplots()
    cmat = confusion_matrix(y_test, y_predict)
    sns.heatmap(cmat, annot=True, fmt="d")
    plt.savefig("Confusion_matrix.png")
    print(f'Confusion_matrix.png saved!\n')
    plt.show()
    # We are showing that we are guessing mostly right but there are times where
    # we make some wrong guesses


    # plotting and checking for collinearity
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), vmin=0.9, vmax=1, ax=ax)
    plt.savefig("Collinearity.png")
    print(f'Collinearity.png saved!\n')
    plt.show()
    # We show no strong correlation between the input features.

    
    # Going to try the SVC estimator to see how well that does.
    svc = SVC()
    svc = svc.fit(X_Outliers_Removed, y_train_Samples_Reduced)
    y_predict = svc.predict(X_test)
    
    print(f'\nSVC Train Score: {svc.score(X_Outliers_Removed, y_train_Samples_Reduced)}')
    print(f'SVC Test Score: {svc.score(X_test, y_test)}\n')
    print(f'r2: {metrics.r2_score(y_test, y_predict)}\n')
    print(f'\nClassification Report:\n{classification_report(y_test, y_predict)}\n')
    # Same Scores and outputs

    # One last thing to try, lets scale the data to see if it helps the score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_Outliers_Removed)

    knn = KNeighborsClassifier()
    knn.fit(X_scaled, y_train_Samples_Reduced)
    y_predict = knn.predict(X_test)

    # Going to print out test and train score
    print(f'\nKNN Train Score: {knn.score(X_scaled, y_train_Samples_Reduced)}')
    print(f'KNN Test Score: {knn.score(X_test, y_test)}\n')
    print(f'r2: {metrics.r2_score(y_test, y_predict)}\n')
    print(f'\nClassification Report:\n{classification_report(y_test, y_predict)}\n')
    # Not much help here, it was just fine unscaled.


    # So overall, we came up with some pretty good scores:
    knn = KNeighborsClassifier()
    knn.fit(X_Outliers_Removed, y_train_Samples_Reduced)
    y_predict = knn.predict(X_test)

    # Going to print out test and train score
    print(f'\nKNN Train Score: {knn.score(X_Outliers_Removed, y_train_Samples_Reduced)}')
    print(f'KNN Test Score: {knn.score(X_test, y_test)}\n')
    print(f'r2: {metrics.r2_score(y_test, y_predict)}\n')
    print(f'\nClassification Report:\n{classification_report(y_test, y_predict)}\n')
    






if __name__ == "__main__":
    Research()