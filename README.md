## American Center Yangon - Deep Learning for Absolute Begineers 
by Arkar Min Aung

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArkAung/dl_acy_workshop)

This repo contains a minimal Flask webserver which you can use to upload an image of handwritten Myanmar digit and get back the predicted digit result. Additionally, publicly accessible URL is created using Ngrok.

The accompanying Jupyter Notebook is meant to be opened in Google CoLab. It contains procedures to:
- Download required libraries
- Augment data
- Build and train nerual network
- Run the webserver with Ngrok

### How to Run
* Create a conda environment: `conda create -n flask_server python=3.7`
* Activate environment: `conda activate flask_server`
* Install requirements: `pip install -r requirements.txt`

### Usage
* `python webserver.py`
* Navigate either to `localhost:5000` or public URL printed in console.
