import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras


# Load your dataset
# Assuming you have a CSV file named 'dataset.csv'
df = pd.read_csv('insta_train.csv')

# Data Preprocessing
X = df[['profile pic','fullname words', 'nums/length fullname',
        'name==username', 'description length', 'external URL', 'private',
        '#posts', '#followers', '#follows']]
y = df['fake']

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize and train an XGBoost classifier
model = XGBClassifier()
model.fit(X, y)

# Initialize and train a Logistic Regression classifier
model1 = LogisticRegression()
model1.fit(X, y)

# Initialize and train a Decision Tree classifier
model2 = DecisionTreeClassifier()
model2.fit(X, y)

# Initialize and train a Support Vector Machine (SVM) classifier
model3 = SVC(kernel='linear')
model3.fit(X, y)

# Initialize and train a Random Forest classifier
model4 = RandomForestClassifier(n_estimators=100, random_state=42)
model4.fit(X, y)

# Initialize an Artificial Neural Network (ANN)
model5 = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model5.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the ANN
model5.fit(X, y, epochs=50, batch_size=32, verbose=1)



# Interactive input from the user
print("Please enter the following information:")
profile_pic = int(input("Profile pic (0 for no pic, 1 for pic): "))
fullname_words = int(input("Full name words count: "))
nums_length_fullname = int(input("Numbers/length of fullname: "))
name_equals_username = int(input("Name == Username (0 for no, 1 for yes): "))
description_length = int(input("Description length: "))
external_url = int(input("External URL (0 for no, 1 for yes): "))
private = int(input("Private account (0 for no, 1 for yes): "))
num_posts = int(input("Number of posts: "))
num_followers = int(input("Number of followers: "))
num_follows = int(input("Number of follows: "))

# Prepare the input data
input_data = [[profile_pic,fullname_words, nums_length_fullname,
               name_equals_username, description_length, external_url, private,
               num_posts, num_followers,num_follows]]

# Standardize the input data
input_data = scaler.transform(input_data)

# Make a prediction
prediction = model.predict(input_data)

# Make a prediction
prediction1 = model1.predict(input_data)

# Make a prediction
prediction2 = model2.predict(input_data)

# Make a prediction
prediction3 = model3.predict(input_data)

# Make a prediction
prediction4 = model4.predict(input_data)

# Make a prediction
prediction5 = model5.predict(input_data)

# Display the result
if prediction[0] == 0:
    print("XGB boost 0")
else:
    print("XGB boost 1")

# Display the result
if prediction1[0] == 0:
    print("Logistic algorithm 0.")
else:
    print("Logistic algorithm 1.")

# Display the result
if prediction2[0] == 0:
    print("Decision tree 0.")
else:
    print("Decision tree 1.")

# Display the result
if prediction3[0] == 0:
    print("SVM ALGORITHM 0.")
else:
    print("SVM algorithm 1.")

# Display the result
if prediction4[0] == 0:
    print("Random forest 0.")
else:
    print("Random forest 1.")

# Display the result
if prediction5[0] > 0.5:
    print("ANN algorithm 0 .")
else:
    print("ANN algorithm 1.")
