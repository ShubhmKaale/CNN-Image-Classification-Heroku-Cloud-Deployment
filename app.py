from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np


app = Flask(__name__)


model_path= 'CNN_Flower_model.h5'
model = load_model(model_path)



dic = {0: 'Marigold', 1: 'Rose', 2: 'Sunflower'}


def predict_label(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(img)
    i = i / 255
    i = np.expand_dims(i, axis=0)
    preds = np.argmax(model.predict(i), axis=1)
    return dic[preds[0]]


@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')


@app.route('/submit', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['my_image']

        img_path = "static/" + f.filename
        f.save(img_path)

        pred = predict_label(img_path, model)
        # return pred
        return render_template("index.html", prediction=pred, img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)
