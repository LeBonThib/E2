import io
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import jsonify, request, Blueprint, render_template, flash
import json

batch_predict = Blueprint('batch_predict', __name__)

model = tf.keras.models.load_model('codebase/model/retinal-oct.h5')

@batch_predict.route('/batch_predict', methods=['GET','POST'])
def batch_predict_panel():
    if request.method == 'POST':
        form_data = request.files
        prediction_result = bulk_infer_image(form_data)
        json_result = json.dumps(prediction_result)
        if not len(json_result) == 0:
            return render_template('index.html', json_result=json_result)
        else:
            return render_template('index.html')
    return 'Bienvenido al circo de los horrores!'

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((150, 150))
    img = np.expand_dims(img, 0)
    img = np.stack((img,)*3, axis=-1)
    return img


def predict_result(img):
    Y_pred = model.predict(img)
    return np.argmax(Y_pred, axis=1)


def bulk_infer_image(form_data):
    if not len(form_data) == 0:
        flash("No file(s) received, please try again.", category='error')
        return render_template('index.html') 

    files = form_data.getlist('images')
    if not files:
        return
    
    batch_result = []
    for file in files:
        img_bytes = file.read()
        img = prepare_image(img_bytes)
        batch_result.append(dict(file=file.filename, prediction=int(predict_result(img))))
    return batch_result


# def bulk_infer_image_api():
#     if 'file' not in request.files:
#         return "No file received, please try again."
    
#     files = request.files.getlist('file')
#     print(files)

#     if not files:
#         return
    
#     batch_result = []
#     for file in files:
#         img_bytes = file.read()
#         img = prepare_image(img_bytes)
#         batch_result.append(dict(file=file.filename, prediction=int(predict_result(img))))

#     return jsonify(batch_result)