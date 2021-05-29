import cv2
import torch
import numpy as np
import torch.nn as nn
from consts import detector_const
from torchvision import transforms
from torch.nn import functional as F
from torchvision.transforms.functional import crop
from torchvision.models.resnet import ResNet, BasicBlock

card_mapping = detector_const.card_mapping
family_mapping = detector_const.family_mapping
rank_mapping = detector_const.rank_mapping
inv_card_mapping = {v: k for k, v in card_mapping.items()}
inv_family_mapping = {v: k for k, v in family_mapping.items()}
inv_rank_mapping = {v: k for k, v in rank_mapping.items()}

class FamilyDetector(nn.Module): 
    def __init__(self):
        super(FamilyDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*11*11, 64)
        self.fc2 = nn.Linear(64, 4)
        
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = self.bn1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = self.bn2(x)
        #x = F.dropout(x, p=0.25)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        #x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        return x

class RankDetector(nn.Module): 
    def __init__(self):
        super(RankDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*11*11, 64)
        self.fc2 = nn.Linear(64, 4)
        
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = self.bn1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = self.bn2(x)
        #x = F.dropout(x, p=0.25)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        #x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        return x

class MNISTResNet(ResNet):
    def __init__(self):
        super(MNISTResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10) # Based on ResNet18
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,bias=False)

class Detector:
    def __init__(self): 
        self.family_detector = FamilyDetector()
        self.family_detector.load_state_dict(torch.load(detector_const.family_detector_path))
        self.family_detector.eval()

        self.rank_detector = RankDetector()
        self.rank_detector.load_state_dict(torch.load(detector_const.rank_detector_path))
        self.rank_detector.eval()

        self.mnist_detector = MNISTResNet()
        self.mnist_detector.load_state_dict(torch.load(detector_const.mnist_detector_path))
        self.mnist_detector.eval()
        
        self.preprocess_family = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda img: crop(img, top=0, left=0, height=75, width=75)),
            transforms.Resize((50,50))
        ])
        
        self.preprocess_rank = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.CenterCrop(110),
            transforms.Resize((50,50))
        ])
        
        self.preprocess_partial_mnist = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize((28,28)),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        
        
    def preprocess_mnist(self, img):
        w = 120
        h = 120
        dsize = (224, 224)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dsize)
        center = img.shape[0] / 2
        x = center - w/2
        y = center - h/2
        crop_img = img[int(y):int(y+h), int(x):int(x+w)]
    
        res = crop_img.copy()
        res[crop_img <= 120] = 250
        res[crop_img > 120] = 0
    
        return self.preprocess_partial_mnist(res)   
            
    
    def prediction_step(self, ROI):
        # ordered from P1 to P4
        cards_played = []
        for img in ROI: 
            pred_family = torch.argmax(self.family_detector(self.preprocess_family(img).unsqueeze(0))).item()
            family = inv_family_mapping.get(pred_family)
            
            pred_rank = torch.argmax(self.rank_detector(self.preprocess_rank(img).unsqueeze(0))).item()
            rank = ''
            if pred_rank == 0:
                pred_mnist = torch.argmax(self.mnist_detector(self.preprocess_mnist(img).unsqueeze(0))).item()
                rank = str(pred_mnist)
            else: 
                rank = inv_rank_mapping.get(pred_rank)
            
            cards_played.append(rank+family)
        return np.array(cards_played)
