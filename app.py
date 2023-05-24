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
 

 
 
@app.route('/upload_csv',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        # Check if CSV files were uploaded
        if 'csvUpload1' in request.files and 'csvUpload2' in request.files:
            file1 = request.files['csvUpload1']
            file2 = request.files['csvUpload2']

            # Check if the filenames are not empty
            if file1.filename != '' and file2.filename != '':
                # Secure the filenames to prevent any malicious file uploads
                filename1 = secure_filename(file1.filename)
                filename2 = secure_filename(file2.filename)
                
                # Save the files to the UPLOAD_FOLDER
                file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
                file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
                
                # Call your function make_dataset with the file paths
                csv_file1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
                csv_file2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
   

if __name__ == "__main__":
    app.run(debug=True)
    
    
