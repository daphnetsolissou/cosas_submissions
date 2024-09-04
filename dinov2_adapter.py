import torch.nn as nn
from transformers import Dinov2Model, Dinov2Config
from mmseg.registry import MODELS

arch_cfg = {'base': {'embed_dim': 768,
                     'num_hidden_layers': 12,
                     'num_attention_heads': 12,
                     'image_size': 518
                     },
            'large': {'embed_dim': 1024,
                      'num_hidden_layers': 24,
                      'num_attention_heads': 16,
                      'image_size': 518
                      }
            }


@MODELS.register_module()
class Dinov2Adapter(nn.Module):

    def __init__(self, arch_name, pretrained=True, patch_size=14, freeze=True, output_layers=None, get_cls=False):
        super().__init__()
        self.arch_name = arch_name
        self.patch_size = patch_size
        self.output_layers = output_layers
        self.get_cls = get_cls

        for key in arch_cfg.keys():
            if self.arch_name.find(key) > -1:
                self.arch_dims = arch_cfg[key]

        if pretrained:
            self.encoder = Dinov2Model.from_pretrained(self.arch_name)
        else:
            config = Dinov2Config(patch_size=self.patch_size, image_size=self.arch_dims['image_size'])
            self.encoder = Dinov2Model(config)

        if freeze:
            self.encoder.requires_grad = False

    def forward(self, x):  # should return a tuple
        # print_image(x)
        encoder_outputs = self.encoder(x, output_hidden_states=True, output_attentions=False)
        height = int(x.shape[2] / self.patch_size)
        width = int(x.shape[3] / self.patch_size)
        if self.output_layers is None:
            return_layers = encoder_outputs.last_hidden_state
            return_layers = return_layers.reshape(-1, height, width, self.arch_dims['embed_dim']).permute(0, 3, 1, 2).contiguous()
        else:
            transformers_hidden_states = encoder_outputs.hidden_states
            return_layers = []
            for layer_idx in self.output_layers:
                if self.get_cls:
                    return_layers.append(transformers_hidden_states[layer_idx][:, 0, :])
                else:
                    layer = transformers_hidden_states[layer_idx][:, 1:, :]
                    layer = layer.reshape(-1, height, width, self.arch_dims['embed_dim']).permute(0, 3, 1, 2).contiguous()
                    return_layers.append(layer)
        return tuple(return_layers)

    def init_weights(self, pretrained=None):
        pass