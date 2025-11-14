# service/app.py
from flask import Flask, request, render_template, jsonify
import joblib
import io
import pandas as pd
from feature_utils import extract_features_from_df

MODEL_PATH = 'models/model_bundle.joblib'

app = Flask(__name__)

model_bundle = joblib.load(MODEL_PATH)
pipeline = model_bundle['pipeline']
le = model_bundle['label_encoder']


@app.route('/')
def index():
    return render_template('index.html')


def parse_csv_bytes(filebytes):
    # tries to read CSV into DataFrame
    s = io.BytesIO(filebytes)
    df = pd.read_csv(s)
    return df


@app.route('/predict', methods=['POST'])
def predict():
    # Accepts multipart form with `file` field or raw CSV body
    if 'file' in request.files:
        f = request.files['file']
        df = parse_csv_bytes(f.read())
    else:
        # raw body
        data = request.data
        if not data:
            return jsonify({'error': 'no data provided'}), 400
        df = parse_csv_bytes(data)

    feats = extract_features_from_df(df)
    # ensure shape
    feats = feats.reshape(1, -1)
    pred_enc = pipeline.predict(feats)
    pred_label = le.inverse_transform(pred_enc)[0]
    resp = {'predicted_zone': str(pred_label)}

    # if request came from form, render front-end
    if request.form.get('from_form') == '1' or request.content_type.startswith('multipart/form-data'):
        return render_template('index.html', prediction=pred_label)
    return jsonify(resp)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
