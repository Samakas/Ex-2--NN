<H3>Name: Samakash R S</H3>
<H3>Register no. 212223230182 </H3>
<H3>Date: 15/11/2026 </H3>
<H3>Experiment No. 2 </H3>


# Implementation of Perceptron for Binary Classification
# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

 


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
# PROGRAM:
   
~~~

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# The dataset has no header, so we define column names manually
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv("/content/pima-indians-diabetes.csv", names=column_names)

# Using 'Age' and 'Glucose' as features, and 'Outcome' (0 or 1) as target
X = df[['Age', 'Glucose']].values
y = df['Outcome'].values

plt.figure(figsize=(8, 5))

plt.scatter(X[y == 0, 0], X[y == 0, 1],
            color='blue', label='No Diabetes')

plt.scatter(X[y == 1, 0], X[y == 1, 1],
            color='red', label='Diabetes')

plt.xlabel("Age")
plt.ylabel("Glucose")
plt.title("Pima Indians Diabetes Dataset")
plt.legend()
plt.show()


scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


Y_train = np.where(y_train == 0, -1, 1)
Y_test = np.where(y_test == 0, -1, 1)


W = np.zeros(X_train.shape[1])
b = 0.0


learning_rate = 0.01
epochs = 100


errors = []

for epoch in range(epochs):
    total_errors = 0

    for i in range(len(X_train)):


        v = np.dot(W, X_train[i]) + b


        t = 1 if v >= 0 else -1


        if t != Y_train[i]:
            W = W + learning_rate * Y_train[i] * X_train[i]
            b = b + learning_rate * Y_train[i]
            total_errors += 1

    errors.append(total_errors)


plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), errors, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Number of Misclassifications")
plt.title("Perceptron Training Error")
plt.grid(True)
plt.show()


def predict(X):
    net = np.dot(X, W) + b
    return np.where(net >= 0, 1, -1)

Y_pred = predict(X_test)


y_pred = np.where(Y_pred == -1, 0, 1)

accuracy = accuracy_score(y_test, y_pred)

print("Final Weight Vector:", W)
print("Final Bias:", b)
print("Number of Epochs:", epochs)
print("Test Accuracy:", accuracy * 100, "%")
~~~
# OUTPUT:

 <img width="877" height="443" alt="image" src="https://github.com/user-attachments/assets/9ff10f7b-e5c2-4f31-b91f-7bf675babd36" />
<img width="831" height="523" alt="image" src="https://github.com/user-attachments/assets/12d0af2d-f285-4834-938e-0a3ff5c6d505" />


# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
