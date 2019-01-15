from api.uploader import Uploader
from api.deep_guided_filter import ImageProcessing
from api.face_swap import FaceSwap

from flask import Flask
from flask_cors import CORS
from flask_restful import Api


app = Flask(__name__)
CORS(app)
deep_api = Api(app)
deep_api.add_resource(Uploader, '/upload')
deep_api.add_resource(ImageProcessing, '/deep_guided_filter/<string:task>')
deep_api.add_resource(FaceSwap, '/face_swap/<string:task>')


if __name__ == '__main__':
    app.run(debug=True)
