from torch import nn


class BengalClassifier(nn.Module):
    def __init__(self, backbone, hidden_size=2560, class_num=168*11*8):
        super(BengalClassifier, self).__init__()
        self.backbone = backbone
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_size, class_num)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, inputs):
        bs = inputs.shape[0]
        feature = self.backbone.extract_features(inputs)
        feature_vector = self._avg_pooling(feature)
        feature_vector = feature_vector.view(bs, -1)
        # returns new tensor with same data but different shape
        feature_vector = self.ln(feature_vector)

        out = self.fc(feature_vector)
        return out