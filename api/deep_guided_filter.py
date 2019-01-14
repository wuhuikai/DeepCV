import os
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from flask import request
from flask_restful import Resource


def to_tensor(x):
    return transforms.ToTensor()(x).unsqueeze(0)


def to_array(tensor):
    return np.squeeze(tensor.data.cpu().numpy()).transpose((1, 2, 0)).tolist()


models = {task: torch.jit.load(os.path.join('models', task, 'hr_net_latest_jit.pth'))
          for task in ['auto_ps', 'l0_smooth', 'multi_scale_detail_manipulation', 'style_transfer', 'non_local_dehazing']}


class ImageProcessing(Resource):
    def get(self, task):
        if task in models:
            path = request.form['path']
            r, eps = int(request.form['r']), float(request.form['eps'])

            im_lr = Image.open(path).convert('RGB')
            im_lr = to_tensor(im_lr)

            A, b = models[task](im_lr, r, eps)
            A, b = to_array(A), to_array(b)

            return {'A': A, 'b': b, 'status': 'success'}

        return {'status': 'error'}
