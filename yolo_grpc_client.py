from libs.grpc.appearance_pb2_grpc import AppearanceStub
from libs.grpc.appearance_pb2 import AppearanceRequest
import numpy as np
import grpc
import pickle
import time
import cv2
from tqdm import tqdm
def is_grpc_ready(channel):
    try:
        grpc.channel_ready_future(channel).result(timeout=60)
        return True
    except grpc.FutureTimeoutError:
        return False

class YoloGrpcClient:
    def __init__(self, ip, port):
        self.channel = grpc.insecure_channel("{}:{}".format(ip, port))
        while not is_grpc_ready(self.channel):
            # Logger.info('- Waiting for GCN grpc server...')
            time.sleep(1)
        # Logger.info('-> GCN grpc server is ready!')
        # print('GCN grpc server is ready!')
        self.yolo_stub = AppearanceStub(self.channel)
        # self.image_size_str = "{}x{}".format(image_size[0], image_size[1])
        # self.image_size_str = image_size

    def predict(self, image):
        '''
        '''
        image_str = pickle.dumps(image)
        request = AppearanceRequest(image=image_str)
        # try:
        response = self.yolo_stub.predict(request)
        bboxes = pickle.loads(response.bboxes)
        probs = pickle.loads(response.probs)
        classes = pickle.loads(response.classes)

        return bboxes, probs, classes


if __name__ == '__main__':
    img = cv2.imread('vlcsnap-2022-09-20-16h07m53s021.png')
    detector = YoloGrpcClient('localhost', 50100)
    for i in tqdm(range(500)):
        boxs, probs, _ = detector.predict(img)
    print(boxs)
