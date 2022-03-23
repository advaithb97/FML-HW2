from sklearn.svm import SVC

import pandas as pd

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
 
from sklearn.utils import shuffle

from sklearn.model_selection import cross_val_score


from matplotlib import pyplot as plt 


abalone = pd.read_csv('abalone.csv')

abalone.columns = ['Sex', 'Length', 'Diameter', 'Height',
                   'Whole weight', 'Shucked weight',
                   'Viscera weight', 'Shell weight', 'Rings']

data_train = abalone.head(3133)

data_test = abalone.tail(1044)

# all attributes except rings from training data
X_train = data_train.iloc[:, 1:8]

#rings attribute from training data
y_train = data_train.iloc[:, -1]

#all attributes except rings from test data
X_test = data_test.iloc[:, 1:8]

#rings attribute from test data
y_test = data_test.iloc[:, -1]
 

#Feature Scaling
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


#print(type(X_train_std))
#print(type(y_train))

y_train = y_train.to_numpy()
y_train_binary = (y_train <= 9).astype(int)
print(y_train.shape)
print(X_train_std.shape)
#get scaled training data all in one dataframe
scaled_train = np.concatenate((X_train_std, y_train_binary[:,None]), axis = 1)



#randomly split training data into 5 equal size sets
#first shuffle the rows of the training data then split
np.random.shuffle(scaled_train)
training_partition = np.array_split(scaled_train, 5)


#binary value of y_test
y_test = y_test.to_numpy()
y_test_binary = (y_test <= 9).astype(int)


#best d-C pair
best_pair = (0, 0)
min_error = 1


# Training a SVM classifier using SVC class
k = 3
for d in range(1, 6):
    avg_cross_errs = []
    for j in range(-k, k+1):
        C = 3**j
        part_test_errors = []
        part_train_errors = []
        for part in training_partition: 
            x_p = part[:, 1:8]
            y_p = part[:, -1]
            svm = SVC(kernel = 'poly', degree = d, C = C)
            svm.fit(x_p, y_p)
            y_pred = svm.predict(X_test_std)
            test_error = np.sum(y_pred != y_test_binary)
            test_error /= 1044
            part_test_errors.append(test_error)
            
            #get training error
            y_train_predict = svm.predict(x_p)
            train_error = np.sum(y_train_predict != y_p)
            train_error /= len(y_p)
            part_train_errors.append(train_error)
            
        avg_error = np.mean(part_test_errors)
        if(avg_error < min_error):
            min_error = avg_error
            best_pair = (C, d)
        #stddev_error = np.std(part_test_errors)
        
        avg_cross_errs.append(avg_error)
        #std_cross_errs.append(stddev_error)
        
        avg_train_error = np.mean(part_train_errors)
    
    #plot Cross validation errors as function of C for each d
    x_vals = [j for j in range(-k, k+1)]
    title = 'Cross Validation Error based on C = 3^j, for d = ' + str(d)
    plt.title(title)
    plt.xlabel("j Values")
    plt.ylabel("Cross Validation Errors")
    #plot cross-validation errors plus or minus one standard deviation
    stdval = np.std(avg_cross_errs)
    plt.errorbar(x_vals, avg_cross_errs, stdval)
    plt.show()

print(best_pair)



C_star = best_pair[0]
d_star = best_pair[1]

avg_cross_errors = []
avg_test_errors = []
avg_support_vecs = []
for d in range(1, 6):
    cross_errs = []
    test_errs = []
    nums_support_vectors = []
    for i in range(len(training_partition)):
        holdout = training_partition[i]
        rest = training_partition.copy()
        rest.pop(i)
        rest = np.concatenate(rest, axis = 0)
        x_p = rest[:, 1:8]
        y_p = rest[:, -1]
        svm = SVC(kernel = 'poly', degree = d, C = C)
        svm.fit(x_p, y_p)
        
        #5-fold cross validation error
        y_train_predict = svm.predict(x_p)
        cross_val_error = np.sum(y_train_predict != y_p)
        cross_val_error /= len(y_p)
        cross_errs.append(cross_val_error)
        
        #test error
        y_pred = svm.predict(X_test_std)
        test_error = np.sum(y_pred != y_test_binary)
        test_error /= 1044
        test_errs.append(test_error)
        
        
        num_support_vecs = np.sum(svm.n_support_)
        nums_support_vectors.append(num_support_vecs)
        
        
    avg_cross = np.mean(cross_errs)
    avg_cross_errors.append(avg_cross)
    avg_test = np.mean(test_errs)
    avg_test_errors.append(avg_test)
    avg_support = np.mean(nums_support_vectors)
    avg_support_vecs.append(avg_support)

#plot d vs avg cross errors, avg test errors, num support vecs
x_vals = np.arange(1,6)
plt.title("5-fold Cross-Validation Errors Based On Degree d")
plt.xlabel("Degree d For Polynomial Kernel (From 1 to 5)")
plt.ylabel("5-fold Cross Validation Errors")
plt.plot(x_vals, avg_cross_errors)
plt.show()

plt.title("Test Errors Based On Degree d")
plt.xlabel("Degree d For Polynomial Kernel (From 1 to 5)")
plt.ylabel("Test Errors")
plt.plot(x_vals, avg_test_errors)
plt.show()


plt.title("Average Number of Support Vectors Based On Degree d")
plt.xlabel("Degree d For Polynomial Kernel (From 1 to 5)")
plt.ylabel("Average Number of Support Vectors")
plt.plot(x_vals, avg_support_vecs)
plt.show()



#for (C,d) equal to (C*, d*), plot train, test errors based on sample
part_test_errors = []
part_train_errors = []
for part in training_partition: 
    x_p = part[:, 1:8]
    y_p = part[:, -1]
    svm = SVC(kernel = 'poly', degree = d_star, C = C_star)
    svm.fit(x_p, y_p)
    y_pred = svm.predict(X_test_std)
    test_error = np.sum(y_pred != y_test_binary)
    test_error /= 1044
    part_test_errors.append(test_error)
    
    #get training error
    y_train_predict = svm.predict(x_p)
    train_error = np.sum(y_train_predict != y_p)
    train_error /= len(y_p)
    part_train_errors.append(train_error)



x_vals = np.arange(1,6)
plt.title("Training Errors for (C*, d*) Based on Training Sample 1-5")
plt.xlabel("Sample Number (from 1 to 5)")
plt.ylabel("Training Errors")
plt.plot(x_vals, part_train_errors)
plt.show()


plt.title("Testing Errors for (C*, d*) Based on Training Sample 1-5")
plt.xlabel("Sample Number (from 1 to 5)")
plt.ylabel("Testing Errors")
plt.plot(x_vals, part_test_errors)
plt.show()