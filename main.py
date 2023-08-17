import numpy
import pandas
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

#Load data
data = pandas.read_csv('diabetes.csv')

def standardization(in_data):
    z = (in_data - (in_data.mean(axis=0)) / (in_data.std(axis=0)))
    return z

#Scaling data
features = data.drop('Outcome', axis=1)
scaled_features = standardization(features)

#Splitting data into training and testing data
print('Splitting data...')
x_train, x_test, y_train, y_test = train_test_split(scaled_features, data['Outcome'], test_size=0.3)

#KNN
print('Calculating K values...')
acc_val = []
k_val = list(range(1, 100))

for k in k_val:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    prediction = knn.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    acc_val.append(accuracy)

best_k_val = k_val[numpy.argmax(acc_val)]
print('Plotting the values...')
plt.plot(k_val, acc_val)
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.show()
print(f'best K value: {best_k_val}')

#5 Fold Cross Validation
cvs = cross_val_score(KNeighborsClassifier(n_neighbors=best_k_val), scaled_features, data['Outcome'], cv=5)
mcv_acc = numpy.mean(cvs)
std_cvs_acc = numpy.std(cvs)
print(f'5 fold cross validation mean accuracy: {mcv_acc}')
print(f'5 fold cross validation st. deviation: {std_cvs_acc}')

#Confusion Matrix
knn_best = KNeighborsClassifier(n_neighbors=best_k_val)
knn_best.fit(x_train, y_train)
y_p_best = knn_best.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_p_best)
print(f'Confusion matrix: {conf_matrix}')

#Leave One Out
loo_score = cross_val_score(KNeighborsClassifier(n_neighbors=best_k_val), scaled_features, data['Outcome'], cv=LeaveOneOut())
mean_loo_acc = numpy.mean(loo_score)
std_loo_acc= numpy.std(loo_score)
print(f'Leave one out mean accuracy: {mean_loo_acc}')
print(f'Leave one out st. deviation: {std_loo_acc}')

#Explanation
print(f'By analyzing the results we found out that the best K value for the KNN model is {best_k_val}. ')
print(f'The models accuracy over 5 fold cross validation is {mcv_acc}. The confusion matrix')
print(f'shows the correct and incorrect predictions. The models accuracy from leave one out cross validation')
print(f'is {mean_loo_acc}')
