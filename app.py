from api.uploader import Uploader
from api.deep_guided_filter import ImageProcessing

from flask import Flask
from flask_restful import Api


app = Flask(__name__)
deep_api = Api(app)
deep_api.add_resource(Uploader, '/upload')
deep_api.add_resource(ImageProcessing, '/<string:task>')

if __name__ == '__main__':
    app.run(debug=True)
