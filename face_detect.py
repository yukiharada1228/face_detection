import logging

import cv2 as cv
import numpy as np

from openvino.inference_engine import IECore

logger = logging.getLogger(__name__)


class FaceDetect:
    
    def __init__(self, model_path):

        # モデルの読み込み
        ie_core = IECore()
        network = ie_core.read_network(model_path + '.xml',
                                       model_path + '.bin',)
        self.exec_network = ie_core.load_network(network=network,
                                            device_name='CPU',
                                            num_requests=0,)
        self.input_name = next(iter(network.input_info))
        self.input_size = network.input_info[self.input_name].input_data.shape
        self.output_name = next(iter(network.outputs))
        output_size = self.exec_network.requests[0].output_blobs[self.output_name].buffer.shape

        self.image = None
        self.faces = []

        logger.debug({'input_size': self.input_size,
                      'output_size': output_size})

    # 顔検出
    def detect(self, image, conf=0.8, num=1):
        _, _, h, w = self.input_size
        input_image = cv.resize(image, (h, w)).transpose((2, 0, 1))[np.newaxis]
        input_data = {self.input_name: input_image}
        output_data = self.exec_network.infer(input_data)[self.output_name]
        faces = []
        for data in np.squeeze(output_data):
            if data[1] == 1 and data[2] > conf:
                xmin = max(0, data[3])
                ymin = max(0, data[4])
                xmax = min(1, data[5])
                ymax = min(1, data[6])
                area = (xmax - xmin) * (ymax - ymin)
                face = [xmin, ymin, xmax, ymax, area,]
                faces.append(face)
        faces.sort(key=lambda face: face[-1], reverse=True)
        self.faces = faces[:num]
        self.image = image
        logger.debug({'action': 'detect',
                      'input_image.shape': input_image.shape,
                      'output_data.shape': output_data.shape,
                      'faces': self.faces,
                      'image.shape': self.image.shape})

    # 描画
    def draw(self):
        output_image = self.image.copy()
        for face in self.faces:
            cv.rectangle(output_image, 
                        (int(face[0] * output_image.shape[1]), int(face[1] * output_image.shape[0])), 
                        (int(face[2] * output_image.shape[1]), int(face[3] * output_image.shape[0])), 
                        color=(0, 255, 255), 
                        thickness=3)
        logger.debug({'action': 'draw',
                      'image.shape': self.image.shape,
                      'output_image.shape': output_image.shape})
        return output_image


if __name__ == '__main__':

    import config

    
    capture = config.capture
    face_detect = config.face_detect

    while capture.isOpened():
        
        # 動画の読み込み
        _, frame = capture.read()
        frame = cv.flip(frame, 1)
        cv.putText(frame, 'Exit with Esc', 
                   (int(0.1 * frame.shape[0]), int(0.1 * frame.shape[1])), 
                   cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
        logger.debug({'frame.shape': frame.shape})

        # 顔検出
        face_detect.detect(frame, conf=0.8, num=1)
        output_frame = face_detect.draw()

        # 画像の表示
        cv.imshow('output_frame', output_frame)
        key = cv.waitKey(1)
        if key == 27:
            break
    capture.release()
    cv.destroyAllWindows()
