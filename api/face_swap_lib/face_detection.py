import cv2
import dlib


detector = dlib.get_frontal_face_detector()
## Face detection
def face_detection(args):
    img = cv2.imread(args['path'])
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    faces = detector(img, 1)
    boxes = [(face.left(), face.top(), face.right(), face.bottom()) for face in faces]

    if len(boxes) > 0:
        return {'boxes': boxes, 'status': 'success'}
    else:
        return {'status': 'error'}
