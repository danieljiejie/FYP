from torch import nn
import timm

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EmotionClassifier, self).__init__()
        self.model = timm.create_model('efficientnet_b2', pretrained=True)
        n_features = self.model.classifier.in_features
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)