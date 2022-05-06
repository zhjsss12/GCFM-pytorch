
from torch import nn
from torch.nn import functional as F
import torch
import torchvision.models as models

class PreTrainedResNet(nn.Module):
    def __init__(self):
        super(PreTrainedResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        modules = list(resnet.children())[:-2]   
        self.resnet = nn.ModuleList(modules)
    def forward(self, image):
        """Extract feature vectors from input images."""
        for i, model in enumerate(self.resnet):
            image = model(image)
            if i == 7:
                features = image
        return features

class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()
        self.q = nn.Sequential(
            nn.Linear(1024, 1024, bias=True),
            nn.LayerNorm(1024),
            nn.ReLU()
        )
        self.k = nn.Sequential(
            nn.Linear(1024, 1024, bias=True),
            nn.LayerNorm(1024),
            nn.ReLU()
        )
        self.v = nn.Sequential(
            nn.Linear(1024, 1024, bias=True),
            nn.LayerNorm(1024),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=2)
        self.linear = nn.Sequential(
            nn.Linear(1024, 256, bias=True),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
    def forward(self, x):
        Q = self.q(x).unsqueeze(2)
        K = self.k(x).unsqueeze(2).transpose(1,2)
        V = self.v(x).unsqueeze(2)
        scale_dot = Q.bmm(K)
        activate_dot = self.softmax(scale_dot)
        mat = activate_dot.bmm(V)
        features = self.linear(mat.squeeze(2))
        return features

class Integrator(nn.Module):
    def __init__(self, sample):
        super(Integrator, self).__init__()
        #  action and face  
        self.clips = sample
        self.action_resnet = PreTrainedResNet()
        self.action_key_frame = nn.Linear(2048,1)
        self.layernore = nn.LayerNorm(sample)
        self.softmax = nn.Softmax(dim=1)
        self.action_attention = nn.Sequential(
            nn.Linear(2048, 512, bias=True),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        self.rnn = nn.LSTM(input_size=2048, hidden_size=2048, batch_first=True)        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.face_resnet = PreTrainedResNet()
        self.face_key_frame = nn.Linear(2048,1)
        self.face_attention = nn.Sequential(
            nn.Linear(2048, 512, bias=True),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        self.rnn2 = nn.LSTM(input_size=2048, hidden_size=2048, batch_first=True)
      
        #  text
        self.lstm = nn.LSTM(input_size=768, hidden_size=768, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(768, 1, bias=True)
        )
        self.text_linear = nn.Sequential(
            nn.Linear(768*20, 512, bias=True),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        
        #  audio
        self.audio_linear = nn.Sequential(         
            nn.LayerNorm(312,elementwise_affine=False),            
            nn.Linear(312,512,bias=True)
        )

        #  cross-modal fusion
        self.concat_weight = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 4, bias=True),
            nn.Softmax(dim=1)
        )
        self.w_attn_1 = nn.Sequential(
            nn.Linear(512, 1, bias=True),
            nn.Softmax(dim = 1)
        )
        self.w_attn_2 = nn.Sequential(
            nn.Linear(512, 1, bias=True),
            nn.Softmax(dim = 1)
        )
        self.w_attn_3 = nn.Sequential(
            nn.Linear(512, 1, bias=True),
            nn.Softmax(dim = 1)
        )
        self.w_attn_4 = nn.Sequential(
            nn.Linear(512, 1, bias=True),
            nn.Softmax(dim = 1)
        )
        self.attn12 = SA()
        self.attn13 = SA()
        self.attn14 = SA()
        self.attn23 = SA()
        self.attn24 = SA()
        self.attn34 = SA()
        self.w_attn_all = nn.Sequential(
            nn.Linear(256*6, 256*6, bias=True),
            nn.Softmax(dim = 1)
        )
        self.w_attn_down = nn.Sequential(
            nn.Linear(256*6, 512, bias=True),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False), 
            nn.Linear(512*5, 512, bias=True),
            nn.LayerNorm(512),
            nn.ReLU()
        )


    def forward(self, video, face, audio, text):        
        #  action and facial representation    
        start = 0
        input_dim = 1
        end = input_dim
        fragments = []
        attention = []

        fragmentsf = []
        attentionf = []
        for i in range(self.clips):
            fragment = video[:, start:end, :, :, :]
            fragment = fragment.squeeze(1)
            fragment = self.action_resnet(fragment)
            fragments.append(fragment)

            fragmentf = face[:, start:end, :, :, :]
            fragmentf = fragmentf.squeeze(1)
            fragmentf = self.face_resnet(fragmentf)
            fragmentsf.append(fragmentf)

            start = end
            end = end + input_dim
        
        for i in range(len(fragments)):
            fragment = fragments[i]
            fragment_att = self.avg_pool(fragment)
            fragment_att = fragment_att.view(fragment_att.size(0),-1)
            fragment_att = fragment_att.unsqueeze(1)
            attention.append(fragment_att)

            fragmentf = fragmentsf[i]
            fragment_attf = self.avg_pool(fragmentf)
            fragment_attf = fragment_attf.view(fragment_attf.size(0),-1)
            fragment_attf = fragment_attf.unsqueeze(1)
            attentionf.append(fragment_attf)
        
        
        attention_cat = torch.cat(attention, dim = 1)  
        attention_frame = self.action_key_frame(attention_cat)
        attention_frame = attention_frame.squeeze(2)
        attention_frame = self.layernore(attention_frame)      
        attention_frame = self.softmax(attention_frame)
        attention_frame = attention_frame.unsqueeze(2)
        attention_cat = attention_frame*attention_cat + attention_cat
        self.rnn.flatten_parameters()
        _, (_, c_n) = self.rnn(attention_cat)       
        c_n = c_n.view(c_n.shape[1], -1)
        out1 = self.action_attention(c_n)


        attention_catf = torch.cat(attentionf, dim = 1)  
        attention_framef = self.face_key_frame(attention_catf)
        attention_framef = attention_framef.squeeze(2)
        attention_framef = self.layernore(attention_framef)
        attention_framef = self.softmax(attention_framef)
        attention_framef = attention_framef.unsqueeze(2)
        attention_catf = attention_framef*attention_catf + attention_catf
        self.rnn2.flatten_parameters()
        _, (_, c_n2) = self.rnn2(attention_catf)      
        c_n2 = c_n2.view(c_n2.shape[1], -1)
        out2 = self.face_attention(c_n2)

        #  textual representation  
        out2d, (_, _) = self.lstm(text)
        outtext = out2d[:,:,0:768] + out2d[:,:,768:768*2]
        attention_text = self.attention(outtext)
        attention_text = attention_text.squeeze(2)
        attention_text = self.softmax(attention_text)
        attention_text = attention_text.unsqueeze(2)
        out3 = attention_text*outtext + outtext
        out3 = out3.view(out3.shape[0], -1)
        out3 = self.text_linear(out3)
        
        #  acoustic representation  
        out4 = self.audio_linear(audio)

        #  cross-modal fusion  
        wout1 = self.w_attn_1(out1)
        wout2 = self.w_attn_2(out2)
        wout3 = self.w_attn_3(out3)
        wout4 = self.w_attn_4(out4)

        concat = torch.cat((wout1, wout2, wout3, wout4), 1)
        concat_weight = self.concat_weight(concat)
        w_1 = concat_weight[:,0].unsqueeze(1)
        w_2 = concat_weight[:,1].unsqueeze(1)
        w_3 = concat_weight[:,2].unsqueeze(1)
        w_4 = concat_weight[:,3].unsqueeze(1)
        out1 = out1*w_1 + out1
        out2 = out2*w_2 + out2
        out3 = out3*w_3 + out3
        out4 = out4*w_4 + out4
        
        out12 = torch.cat((out1, out2), 1)
        out13 = torch.cat((out1, out3), 1)
        out14 = torch.cat((out1, out4), 1)
        out23 = torch.cat((out2, out3), 1)
        out24 = torch.cat((out2, out4), 1)
        out34 = torch.cat((out3, out4), 1)

        out12 = self.attn12(out12)
        out13 = self.attn13(out13)
        out14 = self.attn14(out14)
        out23 = self.attn23(out23)
        out24 = self.attn24(out24)
        out34 = self.attn34(out34)

        out_all = torch.cat((out12, out13, out14, out23, out24, out34), 1)

        attn_out = self.w_attn_all(out_all)
        out_all = attn_out.mul(out_all)+out_all
        out_all = self.w_attn_down(out_all)

        out = torch.cat((out1, out2, out3, out4, out_all), 1)
        out = self.linear(out)

        return out

class GNNLayer(nn.Module):
    """Define one GNN layer"""
    def __init__(self,g, in_dim,out_dim):
        super(GNNLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim,out_dim,bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.norm = nn.LayerNorm(in_dim)

    def edge_attention(self,edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self,edges):
        return {'z':edges.src['z'],'e':edges.data['e']}

    def reduce_func(self,nodes):
        alpha = F.softmax(nodes.mailbox['e'],dim =1)
        h = torch.sum(alpha*nodes.mailbox['z'],dim = 1)
        return {'h':h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class GNNPosLayer(nn.Module):
    """Define the first GNN layer (multi-modalities as input)"""
    def __init__(self,g, in_dim,out_dim):
        super(GNNPosLayer, self).__init__()
        self.fusion = Integrator(10)
        self.g = g
        self.fc = nn.Linear(in_dim,out_dim,bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
       
    def edge_attention(self,edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self,edges):
        return {'z':edges.src['z'],'e':edges.data['e']}

    def reduce_func(self,nodes):
        alpha = F.softmax(nodes.mailbox['e'],dim =1)
        h = torch.sum(alpha*nodes.mailbox['z'],dim = 1)
        return {'h':h}

    def forward(self, video, face, audio, text, positions):
        h = self.fusion(video, face, audio, text)
        h = h + positions
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class TwoLayeredPosGAT(nn.Module):
    """Constructing two-layer gnn"""
    def __init__(self,g, in_dim, hidden_dim, out_dim):
        super(TwoLayeredPosGAT, self).__init__()
        self.layer1 = GNNPosLayer(g, in_dim, hidden_dim)
        self.layer2 = GNNLayer(g, hidden_dim, out_dim)

    def forward(self,video, faces, audio, text,positions):
        h = self.layer1(video, faces, audio, text, positions)
        h = F.elu(h)
        h = self.layer2(h)
        return h      



