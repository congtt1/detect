from os import remove

from utils.main_logger import Logger
from utils.helper import is_grpc_ready
from libs.grpc.pose_pb2_grpc import PoseStub
from libs.grpc.pose_pb2 import PoseRequest
import numpy as np
import grpc
import pickle
import time

import cv2
import numpy as np
from tqdm import tqdm
import argparse
import libs.grpc.action_pb2 as action_pb2
import libs.grpc.action_pb2_grpc as action_pb2_grpc
class GCNGrpcClient:
    def __init__(self, ip, port, image_size):
        self.channel = grpc.insecure_channel("{}:{}".format(ip, port))
        while not is_grpc_ready(self.channel):
            Logger.info('- Waiting for GCN grpc server...')
            time.sleep(1)
        Logger.info('-> GCN grpc server is ready!')
        # print('GCN grpc server is ready!')
        self.stub = action_pb2_grpc.ActionStub(self.channel)
        # self.image_size_str = "{}x{}".format(image_size[0], image_size[1])
        self.image_size_str = image_size

    def predict(self, data):
        '''
        data: pose, shape = (1,20,18,2)
        '''

        # try:
        response = self.stub.predict(action_pb2.ActionRequest(poses=pickle.dumps(data),
                                                            image_size=self.image_size_str,
                                                            camid='cam01',
                                                            reload=False,
                                                            is_normal_update=False
                                                            )
                                    )
        actids = np.frombuffer(response.actids, dtype=np.int).tolist()
        probs = pickle.loads(response.probs)
        check_list = pickle.loads(response.check_list)
        # except Exception as e:
        #     Logger.info("[{}] action service does not ready for using now, please wait ...".format(self.camid))
        #     print(e)
        #     actids = []
        #     probs = []
        #     check_list = []
        return actids, probs, check_list

if __name__ == '__main__':
    
    # from visualize import PoseVisualizer, topology_visualize
    parser = argparse.ArgumentParser(description='AsillaPose Client')

    parser.add_argument('--ip', default="localhost", type=str,
                      help='Ip address of the server')
    parser.add_argument('--port', default=50200, type=int,
                      help='expose port of gRPC server')
    parser.add_argument('--image_size', default='640x360', type=str,
                      help='image_size')

    parser.add_argument('--input', default='', type=str,
                      help='image_size')
    args = parser.parse_args()

    client = GCNGrpcClient(args.ip, args.port, args.image_size)
    pose = np.load('test_npy/Breaking/sankyo_record_2007_2_Breaking_00-06-47_3.npy')
    # pose = np.zeros((1,20,18,2))
    # print(pose.shape)
    # pose[:,:,0] /= 2
    # pose[:,:,1] /= 2
    # # pose = np.array(pose, dtype=np.int0)
    pose = np.reshape(pose, (1,20,18,2))
    print(pose)
    action, prob,_ = client.predict(pose.astype(int))
    print(action, prob)