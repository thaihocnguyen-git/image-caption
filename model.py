import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(DecoderRNN, self).__init__()
        self.embd = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)        
    
    def forward(self, features, captions):
        embeded_input = self.embd(captions) 
        embeded_input = torch.cat((features, embeded_input), dim=1)                        
        hidden, _ = self.lstm(embeded_input)
        output = self.linear(hidden)
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        captions = torch.empty(inputs.shape[0], 1).fill_(0).to(device).to(torch.long)
        
        for i in range(max_len):
            prediction = self.forward(inputs, captions)
            captions = torch.argmax(prediction, dim=-1)
                
        return captions.tolist()[0]
            
        
        
            
    