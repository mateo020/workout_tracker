
# Fitness Tracker Web App


[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0.1-brightgreen)](https://flask.palletsprojects.com/)
[![Sci-kit-Learn](https://img.shields.io/badge/Sci--kit--Learn-0.24.2-orange)](https://scikit-learn.org/)

A fitness tracker web application that helps users track their workouts and predict exercises performed using machine learning.

## Project Overview

The Fitness Tracker Web App is designed to assist fitness enthusiasts in tracking their workouts and gaining insights into their exercise routines. The app utilizes a combination of Python, Flask framework, and machine learning techniques to classify exercises performed by users based on collected sensor data.

## Key Features

- Developed a fitness tracker web application using Python and the Flask framework to provide a user-friendly interface for tracking workouts.
- Implemented a machine learning model using the Sci-kit-Learn library to classify exercises such as bench press, squat, deadlift, and row based on the collected sensor data.
- Integrated the web app with MetaMotion sensors to collect raw accelerometer and gyroscope data during workouts for exercise prediction.
- Enabled users to upload their workout data, which includes the collected sensor data, to the web app for exercise classification and analysis.
- The app provides real-time exercise prediction, allowing users to view the exercises they are performing during their workouts.
- Stored the workout details, including exercise predictions, in user profiles for future reference and analysis.

## Technologies Used

- Python: 3.7+
- Flask: 2.0.1
- Sci-kit-Learn: 0.24.2
- MetaMotion Sensors: Integrated for collecting accelerometer and gyroscope data during workouts.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/fitness-tracker-web-app.git
   ```

2. Change into the project directory:

   ```bash
   cd fitness-tracker-web-app
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the Flask development server:

   ```bash
   flask run
   ```

5. Open your web browser and visit `http://localhost:5000` to access the fitness tracker web app.

## Usage

1. Create an account or log in to your existing account.
2. Connect the MetaMotion sensor to the web app to start collecting workout data.
3. Perform exercises while the sensor collects accelerometer and gyroscope data.
4. Upload the collected data to the web app for exercise prediction.
5. View the predicted exercises and store the workout details in your profile.
6. Analyze your workout history, track progress, and gain insights into your exercise routines.

## License

This project is licensed under the [MIT License](LICENSE).

```

Feel free to modify the content or structure of the README file further to accurately represent your project's features and implementation.
