from libs.appearance.yolov5_module import YoloModule
from concurrent import futures
import time
import numpy as np
import grpc
import pickle

import libs.grpc.appearance_pb2 as appearance_pb2
import libs.grpc.appearance_pb2_grpc as appearance_pb2_grpc
from libs.appearance.models.common import ModelLoader
from libs.appearance.models.torch_utils import select_device

# CFG = CustomConfig.get_instance()
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class YoloDetector(object):
    def __init__(self):
        self.detector = None

        self._load_detector()
    def _load_detector(self):
        self.detector = YoloModule('./yolov5n_face.engine')

    def predict(self, image):
        results = self.detector.detect([image])
        return results

class DetetorServer(appearance_pb2_grpc.AppearanceServicer):
    def __init__(self):
        super(appearance_pb2_grpc.AppearanceServicer, self).__init__()
        self.face_detector = YoloDetector()
    
    def predict(self, request, context):
        image = pickle.loads(request.image)
        results = self.face_detector.detector.detect([image])[0]
        classes = []
        boxs = []
        scores = []

        for res in results:
            xmin, ymin, xmax, ymax, score, clsid = res
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            boxs.append([xmin, ymin, xmax, ymax])
            scores.append(score)
            classes.append(clsid)

        classes = pickle.dumps(classes)
        scores = pickle.dumps(scores)
        boxs = pickle.dumps(boxs)

        return appearance_pb2.AppearanceResponse(classes=classes, bboxes=boxs, probs=scores)




def serve(port):
    # is_activated, _, _, _, _, _ = verify()
    # if not is_activated:
    #     print("Appearance has not been activated!")
    #     return

    print("Starting Appearance server...")
    # max_worker = CFG.appearance.getint("max_worker", 10)
    max_worker = 10
    #weight_path = CFG.appearance.get("model", "")
    #device = select_device(device='0', batch_size=0)
    #model = load_model(weight_path, device)
    # model.share_memory()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_worker))
    appearance_pb2_grpc.add_AppearanceServicer_to_server(DetetorServer(), server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()
    # Logger.info("start ok, [::]:{}".format(port))
    print("start ok, [::]:{}".format(port))

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    server_detect = serve(50100)