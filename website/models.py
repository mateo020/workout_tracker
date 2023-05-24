from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func


class Workout(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    bodyweight = db.Column(db.Numeric)
    name = db.Column(db.String(40))
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    exercises = db.relationship('Exercise')
    
class Exercise(db.Model):
    eid = db.Column(db.Integer, primary_key = True)
    weight = db.Column(db.Numeric)
    reps =  db.Column(db.Numeric)
    sets =  db.Column(db.Numeric)
    workout_id = db.Column(db.Integer, db.ForeignKey('workout.id'))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    workouts = db.relationship('Workout')
 