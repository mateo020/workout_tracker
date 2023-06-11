## Fitness Tracker Web Application

**##Features**
Machine Learning Model: The application incorporates a machine learning model that classifies exercises such as bench press, squat, deadlift, and row. The model is trained on collected sensor data to accurately predict the exercises performed during a workout.

Sensor Data Collection: The web app is integrated with MetaMotion sensors to collect raw accelerometer and gyroscope data during workouts. This data is then used as input for the exercise prediction model.

Workout Tracking: Users can upload their workout data, including the collected sensor data, to the application. The application processes the data, predicts the exercises performed, and stores the workout details in the user's profile.

**Getting Started**
To run the fitness tracker web application locally, follow these steps:

Clone the repository: git clone https://github.com/your-username/fitness-tracker.git
Install the required dependencies: pip install -r requirements.txt
Set up the database: Run the create_database function in __init__.py to create the SQLite database.
Configure the MetaMotion sensors: Connect the sensors to the application and ensure they are properly calibrated.
Start the application: Run python app.py to start the Flask development server.
Access the application: Open your web browser and navigate to http://localhost:5000 to access the fitness tracker web application.
**Usage**
Sign up for an account or log in if you already have one.
Connect and calibrate the MetaMotion sensors.
Start a new workout session and perform your exercises while the sensors collect data.
Upload the workout data using the provided CSV upload form.
The application will process the data, predict the exercises performed, and store the workout details in your profile.
View your workout history and track your progress over time
