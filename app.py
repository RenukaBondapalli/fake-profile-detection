import os
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Retrieve the form data
        profile_pic = int(request.form.get("profile pic"))
        fullname_words = int(request.form.get("fullname words"))
        nums_length_fullname = int(request.form.get("nums/length fullname"))
        name_equals_username = int(request.form.get("name==username"))
        description_length = int(request.form.get("description length"))
        external_url = int(request.form.get("external URL"))
        private = int(request.form.get("private"))
        num_posts = int(request.form.get("#posts"))
        num_followers = int(request.form.get("#followers"))
        num_follows = int(request.form.get("#follows"))

        # Your existing code for model prediction here

        df = pd.read_csv('insta_train.csv')
        
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

        input_data = [[profile_pic, fullname_words, nums_length_fullname,
                       name_equals_username, description_length, external_url, private,
                       num_posts, num_followers, num_follows]]
        
        # Standardize the input data using the same scaler as in your main.py
        input_data = scaler.transform(input_data)

        # Make predictions using all models
        prediction = model.predict(input_data)
        prediction1 = model1.predict(input_data)
        prediction3 = model3.predict(input_data)
        prediction4 = model4.predict(input_data)
        prediction5 = model5.predict(input_data)

        # Determine result messages for each model
        if prediction[0] == 0:
            result_message = "XGBoost: Account is real."
        else:
            result_message = "XGBoost: Account is fake."

        if prediction1[0] == 0:
            result_message1 = "Logistic Regression: Account is real."
        else:
            result_message1 = "Logistic Regression: Account is fake."

        if prediction3[0] == 0:
            result_message3 = "SVM: Account is real."
        else:
            result_message3 = "SVM: Account is fake."

        if prediction4[0] == 0:
            result_message4 = "Random Forest: Account is real."
        else:
            result_message4 = "Random Forest: Account is fake."

        if prediction5[0] > 0.5:
            result_message5 = "Artificial Neural Network: Account is fake."
        else:
            result_message5 = "Artificial Neural Network: Account is real."

        # Determine the final result based on majority voting
        predictions = [prediction[0], prediction1[0], prediction3[0], prediction4[0], prediction5[0]]
        majority_vote = sum(predictions) > len(predictions) / 2

        if majority_vote:
            final_result = "Suspicious Account"
        else:
            final_result = "ACCOUNT IS REAL"

        return render_template("index.html", result_message=result_message, result_message1=result_message1,
                               result_message3=result_message3,
                               result_message4=result_message4, result_message5=result_message5,
                                 final_result=final_result)

    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

