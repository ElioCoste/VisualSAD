import torch

# ResNet18 model architecture
from torchvision.models import resnet18

# VGGish model architecture
from torchvggish import vggish, vggish_inpu



class MainModel(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(MainModel, self).__init__()
        # Initialize the audio encoder (pretrained VGGish model)
        self.audio_encoder = vggish()
        self.audio_encoder.eval()
        
        self.visual_encoder = resnet18(pretrained=False)
        
        # Initialize the visual encoder (ResNet18 model without pretrained weights)        
        self.visual_encoder = resnet18(pretrained=False)
        
        

    def forward(self, audio, video):
        audio_features = self.audio_encoder(audio)
        video_features = self.visual_encoder(video)
        combined_features = torch.cat((audio_features, video_features), dim=1)
        output = self.fc(combined_features)
        return output
