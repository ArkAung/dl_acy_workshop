import base64

from fastai.vision import *
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run
UPLOAD_DIR = 'uploads'
OUPUT_FILE = 'output.png'


def create_folders():
    if not os.path.isdir(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    parseImage(request.get_data())

    # read parsed image back in 8-bit, black and white mode (L)
    im = open_image('uploads/output.png')
    preds_class, preds_idx, preds_output = learn.predict(im)
    class_idx = preds_class.data.item()
    return MAPPING[class_idx]


def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    output_file = '{}/{}'.format(UPLOAD_DIR, OUPUT_FILE)
    with open(output_file, 'wb') as output:
        output.write(base64.decodebytes(imgstr))


if __name__ == "__main__":
    create_folders()

    learn = load_learner('train', 'alphabets.pkl')
    MAPPING = {v: k for k, v in learn.data.c2i.items()}
    app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
    app.secret_key = 'supersecret'
    app.run()
