import argparse
import base64
import os

from fastai.vision import *
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok


app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run
UPLOAD_DIR = 'uploads'
OUPUT_FILE = 'output.png'
TITLE_MAPPING = {'digits': "၁၂၃၄",
                 'alphabets': "ကခ"}
STR_TYPE_MAPPING = {'digits': "နံပါတ်",
                    'alphabets': "အက္ခရာ"}


def get_args():
    parser = argparse.ArgumentParser(description="Web app to recognize hand written Myanmar characters")
    parser.add_argument('--type', type=str, help='Type of recognition [digits|alphabets]')
    parser.add_argument('--weights', type=str, default=None, help='Saved weight file')
    return parser.parse_args()


def create_folders():
    if not os.path.isdir(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)


@app.route('/')
def index():
    return render_template('index.html', title=TITLE_MAPPING[args.type],
                           str_type=STR_TYPE_MAPPING[args.type])


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
    args = get_args()
    assert args.type in ['digits', 'alphabets'], "--type should either be 'digits' or 'alphabets'"

    create_folders()

    if args.weights is None:
        learn = load_learner('train', 'export.pkl')
    else:
        path, filename = os.path.split(args.weights)
        learn = load_learner(path, filename)
    MAPPING = {v: k for k, v in learn.data.c2i.items()}
    app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
    app.secret_key = 'supersecret'
    app.run()
