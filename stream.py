import cv2
import pafy
import numpy as np
import argparse
import torch
from models import TSN
from transforms import *
from PIL import Image
import time


class Queue():
    def __init__(self, size):
        self.size = size
        self.container = []
    
    def enqueue(self, item):
        if len(self.container) < self.size:
            self.container.append(item)
        else:
            print('Buffer full')

    def dequeue(self):
        if not self.isempty():
            self.container.pop(0)
        else:
            print("Buffer empty")

    def get(self):
        return self.container

    def isempty(self):
        return len(self.container) == 0

    def isfull(self):
        return (len(self.container) == self.size)

    def __getitem__(self, i):
        return self.container[i]

    def __len__(self):
        return len(self.container)

def arp(imgs):
    """
    Exact replica of the rank pooling algorithm described in the paper, 
    including the equations and notations used
    args:
        imgs : list of rgb images 
    """

    T = len(imgs)
  
    harmonics = []
    harmonic = 0
    for t in range(0, T+1):
        harmonics.append(harmonic)
        harmonic += 1/(t+1)

    weights = []
    for t in range(1,T+1):
        weight = 2 * (T - t + 1) - (T+1) * (harmonics[T] - harmonics[t-1])
        weights.append(weight)
        
    feature_vectors = []
    for i in range(len(weights)):
        feature_vectors.append(imgs[i] * weights[i])

    feature_vectors = np.array(feature_vectors)
    
    rank_pooled = np.sum(feature_vectors, 0)

    return rank_pooled
    
def main():

    '''
    Use this to apply the streaming operation over a youtube video
    '''
    # url = 'https://www.youtube.com/watch?v=FqJdzYY_Fas'
    # vpafy = pafy.new(url)
    # play = vpafy.getbest(preftype = 'webm')
    # cap = cv2.VideoCapture(play.url)

    '''
    Use this to apply the rank pooling over a local video file
    '''
    cap = cv2.VideoCapture('videodata/savage.mp4')

    buffer = Queue(5)

    model = TSN(2, 1, 'RGB', base_model = 'BNInception')
    checkpoint = torch.load('checkpoints/split_1_rgb_model_best.pth.tar')
    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}

    model.load_state_dict(base_dict)
    input_mean = model.input_mean
    input_std = model.input_std
    input_size = model.input_size
    scale_size = model.scale_size
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3,4]).cuda()
    model.eval()
    
    cropping = torchvision.transforms.Compose([
        GroupScale(scale_size),
        GroupCenterCrop(input_size),
    ])
    
    operations = torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=True),
                       ToTorchFormatTensor(div=False),
                       GroupNormalize(input_mean, input_std),
                   ])

    while True:
        ret, frame = cap.read()

        to_display = cv2.resize(frame, (640, 360))
        to_compute = cv2.resize(frame, (340, 256))

        frame = cv2.cvtColor(to_compute, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
    
        if not buffer.isfull():
            buffer.enqueue(frame)
            continue
        else:
            buffer.dequeue()
            buffer.enqueue(frame)

        frames = buffer.get()
        frames = operations(frames)

        input_var = torch.autograd.Variable(frames)
        output = model(input_var).data.cpu().numpy().copy().mean(axis = 0)

        output_proba = np.exp(output) / np.sum(np.exp(output))

        pred = np.argmax(output)
        
        color = (0,0,255) if pred == 1 else (0,255,0)
        
        aggressiveness = 'Aggressive {:.3f}'.format(output_proba[1]) if pred == 1 else 'Passive {:.3f}'.format(output_proba[0])
        
        cv2.putText(img = to_display, text = aggressiveness, org = (20,20), color = color, fontScale = 0.6, fontFace = cv2.FONT_HERSHEY_SIMPLEX)
        cv2.imshow('frame', to_display)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()