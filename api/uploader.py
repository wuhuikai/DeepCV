import os
import uuid
import werkzeug

from flask_restful import Resource, reqparse


upload_folder = '/tmp/imgs'
if not os.path.isdir(upload_folder):
    os.makedirs(upload_folder)

parser = reqparse.RequestParser()
parser.add_argument('image', type=werkzeug.datastructures.FileStorage, location='files')


class Uploader(Resource):
    def put(self):
        data = parser.parse_args()
        photo = data['image']
        if photo:
            filename = os.path.join(upload_folder, '{}.png'.format(uuid.uuid4()))
            photo.save(filename)
            return {'data': filename, 'status': 'success'}

        return {'status': 'error'}
