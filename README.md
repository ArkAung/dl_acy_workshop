## American Center Yangon - Deep Learning for Absolute Begineers 
by Arkar Min Aung

## For those who are doing workshop

Workshop exercise #1: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArkAung/dl_acy_workshop/blob/master/ACY_Workshop%20(Digits).ipynb)

Workshop exercise #2: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArkAung/dl_acy_workshop/blob/master/ACY_Workshop%20(Alphabets).ipynb) 

## For those who want to test this

This repo contains a minimal Flask webserver will serve you a minimal web app which recognizes Myanmar handwritten
digits and alphabets based on your drawings. Additionally, publicly accessible URL for the webserver is created using Ngrok.

The accompanying Jupyter Notebook is meant to be opened in Google CoLab. It contains procedures to:
- Download required libraries
- Augment data
- Build and train neural network
- Run the webserver with Ngrok

### Preparation
* Create a conda environment: `conda create -n flask_server python=3.7`
* Activate environment: `conda activate flask_server`
* Install requirements: `pip install -r requirements.txt`
* Download weights file: [alphabets](https://drive.google.com/file/d/1hcb7OFhQ5CKVv3M_Ll1xup6KTKI72uvb/view?usp=sharing) | [digits](https://drive.google.com/file/d/1wiB2JSNZHUQn_9dhITtuTSYayAk20Wts/view?usp=sharing)
    * **Note:** The neural networks are trained minimally with very small training dataset. Their performance will be be subpar. 
### How to Run
* `python webserver.py --type [alphabets|digits] --weights /path/of/downloaded/weight/file.pkl`
* Navigate either to `localhost:5000` or public URL printed in console.
