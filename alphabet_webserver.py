import os

from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename

from fastai.vision import *

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run
UPLOAD_DIR = 'uploads'
RESULT_DIR = 'results'


def create_folders():
    if not os.path.isdir(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)


@app.route('/')
def index():
    return render_template('index_alpha.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'GET':
        return render_template('upload_alpha.html')
    elif request.method == 'POST':
        if 'file' not in request.files:
            return jsonify(success=False,
                           description="No file part")
        file = request.files['file']
        if file.filename == '':
            return jsonify(success=False,
                           description="No file selected for uploading")
        if file:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            im = open_image(upload_path)
            preds_class, preds_idx, preds_output = learn.predict(im)
            class_idx = preds_class.data.item()
            return jsonify(success=True,
                           result=MAPPING[class_idx])


if __name__ == "__main__":
    create_folders()

    learn = load_learner('train', 'export.pkl')
    MAPPING = {v:k for k,v in learn.data.c2i.items()}
    app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
    app.secret_key = 'supersecret'
    app.run()