from .a2rl_lib.A2RL import auto_cropping

from flask_restful import Resource, reqparse


class AutoCrop(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('path', type=str)

    def get(self):
        args = AutoCrop.parser.parse_args()
        box = auto_cropping(args['path'])

        return {'box': box, 'status': 'success'}
