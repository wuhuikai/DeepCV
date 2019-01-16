from .blending_lib.run_gp_gan import blending

from flask_restful import Resource, reqparse


class Blending(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('color_weight', type=float, default=1)
    parser.add_argument('src', type=str)
    parser.add_argument('dst', type=str)
    parser.add_argument('mask', type=str)

    def get(self):
        args = Blending.parser.parse_args()
        return blending(args)
