from mmengine.model import BaseModel
from mmengine.registry import MODELS
import torch
from transformers import Dinov2Model, Dinov2Config
import torch.nn.functional as F


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=2):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1, 1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        return self.classifier(embeddings)


@MODELS.register_module(name='Dinov2LRP')
class Dinov2LRP(BaseModel):
    def __init__(self, model_name, load_pretrained_weights=True, freeze_encoder=True, embed_dim=768, patch_size=14,
                 resH=32, resW=32, num_labels=2):
        super(Dinov2LRP, self).__init__()

        if load_pretrained_weights:
            self.encoder = Dinov2Model.from_pretrained(model_name)
        else:
            config = Dinov2Config(patch_size=patch_size, image_size=518)
            self.encoder = Dinov2Model(config)

        if freeze_encoder:
            self.encoder.requires_grad = False
        self.classifier = LinearClassifier(embed_dim, resH, resW, num_labels=num_labels)

    def forward(self, imgs, data_samples=None, mode='tensor'):
        encoder_outputs = self.encoder(imgs, output_hidden_states=False, output_attentions=False)
        last_hidden_states = encoder_outputs.last_hidden_state
        # print(last_hidden_states.shape)
        patch_embeddings = last_hidden_states[:, 1:, :]
        # print(patch_embeddings.shape)
        logits = self.classifier(patch_embeddings)
        # print(logits.shape)
        logits = torch.nn.functional.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
        # print(logits.shape)
        # print(data_samples['labels'].unsqueeze(1).shape)
        if mode == 'loss':
            ground_truth = data_samples['labels'].float()
            return {'loss': F.binary_cross_entropy_with_logits(logits, ground_truth.unsqueeze(1))}
        elif mode == 'predict':
            return logits, data_samples
        elif mode == 'tensor':
            return logits
