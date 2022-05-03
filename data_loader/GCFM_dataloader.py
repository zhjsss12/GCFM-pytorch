from torchvision import transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import torch
import os
import numpy as np
import math
from config.fold_setting import *
from config.basic_config import *
from utils.audio_features_extractor import AudioFeatureExtractor
from bert_serving.client import BertClient

class GCFMLoader():
    def __init__(self,):
        super(GCFMLoader,self).__init__()     
        self.audio_extractor = AudioFeatureExtractor()    
        self.opt = DefaultConfig()
        self.fold_set = FoldConfig() 
   
    def load_edges(self, path):    
        f = open(path,'r',encoding = 'utf8').readlines()
        edges = [(int(i.strip().split()[0]), int(i.strip().split()[1])) for i in f]
        return edges
    
    def PosEmbedding(self, position, d_model):
        pe = torch.zeros(d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pe[0::2] = torch.sin(position * div_term)
        pe[1::2] = torch.cos(position * div_term)
        return pe

    def load_vat_features(self, video_path, face_path, audio_path, text_path, sample, fold, cropsize, sr):
        print("Load process init ...")
        action_transform = T.Compose([
                T.RandomResizedCrop(cropsize),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.11576645, 0.10429395, 0.112622395), 
                            (0.21120705, 0.18992329, 0.19986944))  
            ])

        facial_transform = T.Compose([
                T.RandomResizedCrop(cropsize),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.39884713, 0.2939282, 0.31120068), 
                            (0.19167131, 0.14814788, 0.15341194)) 
            ])
        # The above two sets of Normalize parameters were obtained from Live-Action Program Dataset

        bc = BertClient(port=5777,port_out=5778)
        print("Init finish .")
        print("Start Data loading ...")
        Frames = {}
        clips = []
        videos = []
        Y = []
        ids = os.listdir(video_path)
        for id in ids:
            id_path = os.path.join(video_path,id)
            items = os.listdir(id_path)
            for item in items:
                key = int(item.split('.')[0].split('_')[-1])
                Frames[key] = []
                frame_path = os.path.join(id_path, item)
                frames = os.listdir(frame_path)
                pre_names = frames[0].split('.')[0].split('_')
                pre_name = pre_names[0]+'_'+pre_names[1]+'_'+pre_names[2]+'_'+pre_names[3]+'_'
                for i in range(len(frames)):
                    Frames[key].append(os.path.join(frame_path, pre_name+str(i)+'.jpg'))
        
        for key in sorted(Frames.keys()):
            clips.append(Frames[key])
            label = Frames[key][0].split('_')[-4]
            if label == 'truth':
                Y.append(1)
            else:
                Y.append(0)
        

        for clip in clips:
            num_clip = len(clip)
            interval = int(num_clip/sample)
            video = []
            for i in range(sample):
                image = Image.open(clip[i*interval]).convert('RGB')
                image = action_transform(image)
                image = image.unsqueeze(0)
                video.append(image)
            t_video = torch.cat(video,dim=0)
            t_video = t_video.unsqueeze(0)
            videos.append(t_video)
            
        Pics = {}
        pics = []
        faces = []
        ids = os.listdir(face_path)
        for id in ids:
            id_path = os.path.join(face_path,id)
            items = os.listdir(id_path)
            for item in items:
                key = int(item.split('.')[0].split('_')[-1])
                Pics[key] = []
                frame_path = os.path.join(id_path, item)
                frames = os.listdir(frame_path)
                pre_names = frames[0].split('.')[0].split('_')
                pre_name = pre_names[0]+'_'+pre_names[1]+'_'+pre_names[2]+'_'+pre_names[3]+'_'
                for i in range(len(frames)):
                    Pics[key].append(os.path.join(frame_path, pre_name+str(i)+'.jpg'))
        
        for key in sorted(Pics.keys()):
            pics.append(Pics[key])
            
        for clip in pics:
            num_clip = len(clip)
            interval = int(num_clip/sample)
            video = []
            for i in range(sample):
                image = Image.open(clip[i*interval]).convert('RGB')
                image = facial_transform(image)
                image = image.unsqueeze(0)
                video.append(image)
            t_video = torch.cat(video,dim=0)
            t_video = t_video.unsqueeze(0)
            faces.append(t_video)

        Audios = {}
        audios = []
        ids = os.listdir(audio_path)
        for id in ids:
            id_path = os.path.join(audio_path,id)
            items = os.listdir(id_path)
            for item in items:
                key = int(item.split('.')[0].split('_')[-1])
                Audios[key] = os.path.join(id_path,item)
                        
        for key in sorted(Audios.keys()):
            f_path = Audios[key]
            X, sample_rate = self.audio_extractor.read_audio(f_path, sr)
            feature = self.audio_extractor.get_features(X, sample_rate)
            features = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
            audios.append(features)

        Transcriptions = {}
        sentences = []
        ids = os.listdir(text_path)
        for id in ids:
            id_path = os.path.join(text_path,id)
            items = os.listdir(id_path)
            for item in items:
                key = int(item.split('.')[0].split('_')[-1])
                Transcriptions[key] = os.path.join(id_path,item)
                        
        for key in sorted(Transcriptions.keys()):
            f_path = Transcriptions[key]
            sentence = ''
            with open(f_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    sentence = sentence + line
            sentences.append(sentence)

        idx = self.fold_set.idx
        id2index = self.fold_set.id2index

        train_idx = []
        test_idx = []

        for id in idx.keys():
            if id == fold:
                for ids in idx[id]:
                    test_idx += id2index[ids]
            else:
                for ids in idx[id]:
                    train_idx += id2index[ids]

        
        positions = []
        for pp in sorted(id2index.keys()):
            id_item = id2index[pp]
            max_len = len(id_item)
            for i in range(max_len):
                position = self.PosEmbedding(i, 512).unsqueeze(0)
                positions.append(position)
        
        videos = torch.cat(videos, dim=0)
        faces = torch.cat(faces, dim=0)
        audios = torch.cat(audios, dim=0)
        vec_text = bc.encode(sentences)
        vec_text = torch.tensor(vec_text)
        Y = np.asarray(Y)
        
        return videos, faces, audios, vec_text, train_idx, test_idx, torch.cat(positions, dim=0), Y
