from flask import Flask, request, render_template, flash, redirect, url_for,session
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager
from werkzeug.utils import secure_filename
import os

from website.data_cleaning import *
db = SQLAlchemy()
DB_NAME = "database.db"
ALLOWED_EXTENSIONS = set(['csv'])




def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    from .models import User, Workout, Exercise
    
    with app.app_context():
        db.create_all()

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))
    
    UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    
    @app.route('/upload_csv',  methods=("POST", "GET"))
    def uploadFile():
        
        if request.method == 'POST':
            # Check if CSV files were uploaded
            if 'csvUpload1' in request.files and 'csvUpload2' in request.files:
                file1 = request.files['csvUpload1']
                file2 = request.files['csvUpload2']

                # Check if the filenames are not empty
                if file1.filename != '' and file2.filename != '' and allowed_file(file2.filename) and allowed_file(file1.filename):
                    # Secure the filenames to prevent any malicious file uploads
                    filename1 = secure_filename(file1.filename)
                    filename2 = secure_filename(file2.filename)
                    
                    
                    # Save the files to the UPLOAD_FOLDER
                    file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
                    file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
                    
                    
                    # Call your function make_dataset with the file paths
                    csv_file1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
                    csv_file2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
                    
                    
                    acc_df= pd.read_csv(csv_file1)
                    gyr_df= pd.read_csv(csv_file2)

                    data_resampled = make_dataset(csv_file1,csv_file2)   
                    predictions = filter_out_noice(data_resampled)    
                    exercise_names = predicted_exercises(predictions) 
                    print(exercise_names)
                    # Storing exercise_names in session
                    session['exercise_names'] = exercise_names
                    
        flash("Data uploaded!", "success")
        # return render_template("create_post.html", exercise_names=exercise_names)
        return redirect(url_for('views.new_workout'))
    
    return app


def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!------------------------')