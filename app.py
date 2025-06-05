from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from flask_cors import CORS, cross_origin
from ai_utils.utils import decodeImage
from prediction.predict import DogCat

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

# Configure static folder
STATIC_FOLDER = 'static'
app.config['STATIC_FOLDER'] = STATIC_FOLDER

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = DogCat(self.filename)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)
        result = clApp.classifier.predictiondogcat()
        return jsonify([result])
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route("/get_similar_images", methods=['POST'])
@cross_origin()
def get_similar_images():
    try:
        num_images = int(request.json.get('num_images', 5))
        predicted_label, similar_images = clApp.classifier.get_similar_images(num_images)
        
        # Convert paths to be web-accessible
        web_images = []
        for img_path in similar_images:
            # Make sure paths are relative to static folder
            if img_path.startswith('/content/'):
                # Remove /content/ prefix if present
                web_path = img_path.replace('/content/', '')
            else:
                web_path = img_path
            
            web_images.append(web_path)
        
        return jsonify({
            'success': True,
            'prediction': predicted_label,
            'similar_images': web_images
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400

@app.route('/static/<path:filename>')
@cross_origin()
def serve_static(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=5000, debug=True)