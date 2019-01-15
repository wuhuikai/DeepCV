from .face_swap_lib.face_detection import face_detection
from .face_swap_lib.face_swap import face_swap

from flask_restful import Resource, reqparse


class FaceSwap(Resource):
    detection_parser = reqparse.RequestParser()
    detection_parser.add_argument('path', type=str)

    swap_parser = reqparse.RequestParser()
    swap_parser.add_argument('src', type=str)
    swap_parser.add_argument('dst', type=str)
    swap_parser.add_argument('src_box', type=str)
    swap_parser.add_argument('dst_box', type=str)
    swap_parser.add_argument('warp_2d', type=str)
    swap_parser.add_argument('correct_color', type=str)
    tasks = {'detection': (detection_parser, face_detection),
             'face_swap': (swap_parser, face_swap)}

    def get(self, task):
        if task in FaceSwap.tasks:
            parser, api = FaceSwap.tasks[task]

            return api(parser.parse_args())

        return {'status': 'error'}
