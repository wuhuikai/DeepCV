import os
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from flask import request
from flask_restful import Resource

def to_tensor(x):
    return transforms.ToTensor()(x).unsqueeze(0)

def to_img(tensor):
    return Image.fromarray(np.asarray(np.clip(np.squeeze(tensor.data.cpu().numpy()) * 255, 0, 255), dtype=np.uint8).transpose((1, 2, 0)))

models = {task: torch.jit.load(os.path.join('models', task, 'hr_net_latest_jit.pth'))
          for task in ['auto_ps', 'l0_smooth', 'multi_scale_detail_manipulation', 'style_transfer', 'non_local_dehazing']}

class ImageProcessing(Resource):
    def put(self, task):
        if task in models:
            path = request.form['path']
            r, eps = int(request.form['r']), float(request.form['eps'])

            im_hr = Image.open(path).convert('RGB')
            im_lr = transforms.Resize(64, interpolation=Image.NEAREST)(im_hr)
            im_hr = to_tensor(im_hr)
            im_lr = to_tensor(im_lr)

            name = os.path.splitext(os.path.basename(path))[0]
            save_name = '{}_{}_{}_{}.png'.format(name, task, r, eps)
            save_path = os.path.join('static', 'images', save_name)

            image = to_img(models[task](im_lr, im_hr, r, eps))
            if task == 'style_transfer':
                image = image.convert('L')
            image.save(save_path)

            return {'data': save_path, 'status': 'success'}

        return {'status': 'error'}
