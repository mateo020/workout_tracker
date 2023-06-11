from website import create_app
from flask import Flask, render_template, request, session
import pandas as pd
import os
from werkzeug.utils import secure_filename
app = create_app()


# Define folder to save uploaded files to process further
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
 
# Define allowed files (for this example I want only csv file)
ALLOWED_EXTENSIONS = {'csv'}
 
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 

 
 
if __name__ == "__main__":
    app.run(debug=True)
    
    
