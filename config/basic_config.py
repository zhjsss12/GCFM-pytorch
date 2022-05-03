class DefaultConfig():
    def __init__(self):
        self.model_path = './checkpoints' 
        self.graphnode = 334
        self.edgefile = './edges_odd.txt'
        self.batch_size = 32
        self.num_epochs = 80
        self.num_workers = 2
        self.hidden_dim = 512 
        self.graph_indim = 512
        self.class_num = 2
        
class TextConfig(DefaultConfig):
    def __init__(self): 
        super(TextConfig, self).__init__()
        self.seq_len = 20
        self.path = './LiveActionProgram/Transcription'
        
class VideoConfig(DefaultConfig):
    def __init__(self): 
        super(VideoConfig, self).__init__()
        self.sample = 10
        self.crop_size = 64
        self.path = './LiveActionProgram/Frames'
           
class AudioConfig(DefaultConfig):
    def __init__(self): 
        super(AudioConfig, self).__init__()
        self.sr = 30000
        self.path = './LiveActionProgram/Audios'
        
class FaceConfig(DefaultConfig):
    def __init__(self): 
        super(FaceConfig, self).__init__()
        self.sample = 10
        self.crop_size = 64
        self.path = './LiveActionProgram/Faces'
        
