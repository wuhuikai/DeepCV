from api.uploader import Uploader
from api.deep_guided_filter import ImageProcessing
from api.face_swap import FaceSwap
from api.auto_crop import AutoCrop
from api.blending import Blending

from flask import Flask
from flask_cors import CORS
from flask_restful import Api
from flask_compress import Compress

from flask import send_from_directory

app = Flask(__name__)
CORS(app)
Compress().init_app(app)
deep_api = Api(app)
deep_api.add_resource(Uploader, '/upload')
deep_api.add_resource(ImageProcessing, '/deep_guided_filter/<string:task>')
deep_api.add_resource(FaceSwap, '/face_swap/<string:task>')
deep_api.add_resource(AutoCrop, '/auto_crop')
deep_api.add_resource(Blending, '/blending')

@app.route('/images/<string:name>')
def get_image(name):
    return send_from_directory('/tmp/imgs', name, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8888', threaded=True)
