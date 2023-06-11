from flask import Blueprint, render_template, request, flash, jsonify, redirect,url_for,session
from flask_login import login_required, current_user
from . import db
from .models import User, Workout, Exercise
from website.data_cleaning import *
import os
import joblib
import pickle
from werkzeug.utils import secure_filename


model, ref_cols, target = joblib.load("/Users/mateo/OneDrive/Desktop/tracking-barbell-exercises/models/model.pkl")
views = Blueprint('views', __name__) #blueprint for flask application
@views.route('/')
@login_required
def home():
    return render_template("home.html")



@views.route('/new_workout', methods=['GET', 'POST'])
@login_required
def new_workout():
    if request.method == 'POST':
        exercise_names = session.pop('exercise_names', [])
        
        # acc_df= pd.read_csv("../staticFiles/uploads/clusterdata.csv")
        # gyr_df= pd.read_csv("../staticFiles/uploads/clusterdata.csv")

        # data_resampled = make_dataset(acc_df,gyr_df)   
        # predictions = filter_out_noice(data_resampled)    
        # exercise_names = predicted_exercises(predictions) 
        
        
        exercise_names = request.form.getlist('exercise[]')
        sets = request.form.getlist('sets[]')
        reps = request.form.getlist('reps[]')
        pounds = request.form.getlist('pounds[]')
        workout_name = request.form['workout_name']
        bodyweight = request.form['bodyweight']
        
        if len(workout_name) < 1:
            flash('Enter workout name')
            return redirect(url_for('views.new_workout'))

        new_workout = Workout(user_id=current_user.id, name=workout_name, bodyweight=bodyweight)
        db.session.add(new_workout)
        db.session.flush()

        for exercise_name, set_count, rep_count, pound_count in zip(exercise_names, sets, reps, pounds):
            print(f"Exercise: {exercise_name}, Sets: {set_count}, Reps: {rep_count}, Pounds: {pound_count}")
            print("____________________")
            exercise = Exercise(
                weight=pound_count,
                reps=rep_count,  # Access individual element from the list
                sets=set_count,
                workout_id=new_workout.id
            )
            db.session.add(exercise)
        
        db.session.commit()

        flash("Workout submitted successfully!", "success")
        return redirect(url_for('views.home'))

    return render_template("create_post.html")
