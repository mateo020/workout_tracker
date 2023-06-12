# Fitness Tracker Web App

This project is a fitness tracker web application developed using Python and Flask framework. The application utilizes machine learning techniques, feature engineering, and sensor data to track and classify exercises performed during workouts.

## Features

- **Exercise Classification**: The web app includes a machine learning model trained on collected sensor data. This model can classify exercises such as bench press, squat, deadlift, and row, based on the input data.

- **Sensor Data Integration**: The application integrates with MetaMotion sensors to collect raw accelerometer and gyroscope data during workouts. This data is then used for exercise prediction and analysis.

- **Workout Data Upload**: Users can upload their workout data to the web app. The collected data is processed and analyzed to predict the exercises performed during the workout session.

- **User Profiles**: The web app allows users to create profiles and store their workout details. The predicted exercises and other relevant information are saved in the user's profile for tracking and reference.

## Technologies Used

- Python
- Flask
- Sci-kit Learn
- HTML
- Feature Engineering

## Getting Started

To run the fitness tracker web app locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/fitness-tracker-web-app.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Open your web browser and access the application at `http://localhost:5000`.

## Usage

1. Create a user profile by signing up on the web app.

2. Connect the MetaMotion sensor to your device and start a workout session.

3. Upload the collected sensor data to the web app.

4. The application will analyze the data and predict the exercises performed during the workout.

5. View the workout details and exercise predictions in your user profile.
