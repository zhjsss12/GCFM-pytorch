import torch
from data_loader.GCFM_dataloader import GCFMLoader
from config.basic_config import * 
import dgl
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import warnings
from torch.nn import functional as F
import numpy as np
import os
from model.model_GCFM_v2 import TwoLayeredPosGAT
warnings.filterwarnings("ignore")
device = torch.device('cpu')  

def train_graph(fold):
    g = dgl.DGLGraph().to(device)
    opt = DefaultConfig()
    
    g.add_nodes(opt.graphnode)
    loader = GCFMLoader()
    edges = loader.load_edges(opt.edgefile)
    src, dst = tuple(zip(*edges))
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    g.add_edges(g.nodes(), g.nodes())  

    
    graph_indim = opt.graph_indim   
    audio_opt = AudioConfig()
    video_opt = VideoConfig()
    text_opt = TextConfig()
    face_opt = FaceConfig()
    video, faces, audio, text, train_idx, test_idx, positions, labels = loader.load_vat_features(video_opt.path, face_opt.path, audio_opt.path, text_opt.path, video_opt.sample, fold, video_opt.crop_size, audio_opt.sr)
    video = video.to(device)
    faces = faces.to(device)
    audio = audio.to(device)
    text = text.to(device)
    print("Finish load dataset")
    
    train_num = len(train_idx)
    test_num = len(test_idx)
    labels = torch.tensor(labels).to(device)
    labels= labels.type(torch.LongTensor)
    train_idx = torch.tensor(train_idx).to(device)
    test_idx = torch.tensor(test_idx).to(device)
    
    loss_fcn = torch.nn.CrossEntropyLoss()

    model = TwoLayeredPosGAT(g,
        in_dim=graph_indim,
        hidden_dim=opt.hidden_dim,
        out_dim=opt.class_num)

    model.to(device)
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 0.001}
    ],
        lr = 0.005,
        weight_decay = 0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5) 
    
    dur = []
    best_acc = 0
    pre_val, rec_val, fmeasure_val = 0, 0, 0
    best_true_val, best_pred_val = [], []
    for epoch in range(opt.num_epochs):
        if epoch >= 3:
            t0 = time.time()

        logits = model(video, faces, audio, text, positions)
        logp = F.log_softmax(logits, 1)    
        loss = loss_fcn(logp[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch >= 3:
            dur.append(time.time() - t0)
        
        train_acc = torch.sum(logp[train_idx].argmax(dim=1) == labels[train_idx])
        train_acc = train_acc.item() / train_num

        model.eval()
        with torch.no_grad():
            val_pred = []
            val_true = []
            logits = model(video, faces, audio, text, positions)
            logp = F.log_softmax(logits, 1)
            test_acc = torch.sum(logp[test_idx].argmax(dim=1) == labels[test_idx])
            for pred in logp[test_idx].argmax(dim=1):
                val_pred.append(pred)
            for sent in labels[test_idx]:
                val_true.append(sent)
            val_pred = np.asarray(val_pred).astype(int)
            val_true = np.asarray(val_true).astype(int)
            precision = precision_score(val_true, val_pred, average='macro', zero_division=0)*100
            recall = recall_score(val_true, val_pred, average='macro', zero_division=0)*100
            f_measure = f1_score(val_true, val_pred, average='macro', zero_division=0)*100
            test_acc = test_acc.item() / test_num
            
            if test_acc > best_acc and epoch > 20:
                torch.save({'net': model.state_dict()}, os.path.join(opt.model_path, fold+'-GCFM-'+str(epoch)+'.ckpt'))
            if test_acc > best_acc:
                best_acc = test_acc
                pre_val = precision
                rec_val = recall
                fmeasure_val = f_measure
        print("Epoch {:05d} | train Loss {:.4f} | Time(s) {:.4f} train acc {:.4f}".format(
            epoch, loss.item(), np.mean(dur),train_acc))
 
    return best_acc*100, pre_val, rec_val, fmeasure_val
         
            

def main():
    folds = [
        'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9'
    ]
    
    acc_all = 0
    f1_all = 0
    pre_all = 0
    rec_all = 0
    
    for i in range(len(folds)):
        print("##      FOLD  {}     ##".format(folds[i]))
        accuracy_val, precision_val, recall_val, f1_val, = train_graph(folds[i])
        accuracy_val += acc_all
        precision_val += pre_all
        recall_val += rec_all
        f1_val += f1_all

    print('10-Folds Avg Test : Accuracy: {:.4f} %, F1-measure: {:.4f} %, Precision: {:.4f} %, Recall: {:.4f} %'
                .format(acc_all/10, pre_all/10, rec_all/10, f1_all/10))
        
    

if __name__ == '__main__':
    main()
