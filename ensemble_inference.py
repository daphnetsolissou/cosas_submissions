import os
import os.path as osp

import SimpleITK

from mmseg.apis import inference_model, init_model
import numpy as np
import sys

import torch
import cv2

from dinov2LinearProb import Dinov2LRP
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from mmengine.runner import Runner


val_transforms = A.Compose([A.Resize(height=224, width=224),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2()])


def read(path):
    image = SimpleITK.ReadImage(path)
    return SimpleITK.GetArrayFromImage(image)


def write(path, array):
    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(image, path, useCompression=False)


def load_model_checkpoint(checkpoint_path, pretrained_encoder_name, work_dir, device='cpu'):
    model = Dinov2LRP(model_name=pretrained_encoder_name, load_pretrained_weights=False, freeze_encoder=True,
                      embed_dim=768, patch_size=14, resH=16, resW=16, num_labels=1)

    runner = Runner(model=model, work_dir=work_dir)
    runner.load_checkpoint(checkpoint_path, map_location=device)

    model.eval()
    model.to(device)
    return model


def load_mmseg_model(config, checkpoint, device='cpu'):
    model = init_model(config, checkpoint, device=device)
    model.eval()
    return model


def load_lp_model(pretrained_encoder_name, checkpoint, work_dir='temp_dir', device='cpu'):
    model = load_model_checkpoint(checkpoint_path=checkpoint, pretrained_encoder_name= pretrained_encoder_name,
                                  work_dir=work_dir, device=device)
    return model


def predict_mmseg(model, image):
    pred_logits = inference_model(model, image).seg_logits.data
    pred_probs = torch.sigmoid(pred_logits)
    result = pred_probs.squeeze().cpu().numpy()
    return result


def predict_lp(model, image, device, tile_size=256):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tiling_percentage_h = np.round(image.shape[0] / tile_size, 1)
    tiling_percentage_w = np.round(image.shape[1] / tile_size, 1)
    percentages = (tiling_percentage_h, tiling_percentage_w)
    pred_logits = torch.from_numpy(sliding_predict(model, image, device=device,
                                                   tiling_percentages=percentages))
    pred_probs = torch.sigmoid(pred_logits).cpu().numpy()
    return pred_probs


def load_models(models_dict):
    loaded_models = {}
    for key, values in models_dict.items():
        if key.find('lp') > -1:
            model = load_lp_model(**values)
            loaded_models[key] = model
        elif key.find('deeplabv3plus') > -1:
            model = load_mmseg_model(**values)
            loaded_models[key] = model
        else:
            print('Unsupported model')
            sys.exit(0)
    return loaded_models


def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img


def sliding_predict(model, np_image, overlap=None, tiling_percentages=None, device='cpu'):
    np_image_size = np_image.shape

    if tiling_percentages is None:
        tiling_percentages = (2.5, 2.5)
    tile_size = (int(np_image_size[0] // tiling_percentages[0]), int(np_image_size[1] // tiling_percentages[1]))

    if overlap is None:
        overlap = 1 / 3

    stride = np.ceil(tile_size[0] * (1 - overlap))

    num_rows = int(np.ceil((np_image_size[0] - tile_size[0]) / stride) + 1)
    num_cols = int(np.ceil((np_image_size[1] - tile_size[1]) / stride) + 1)
    total_predictions = np.zeros((np_image_size[0], np_image_size[1]))
    count_predictions = np.zeros((np_image_size[0], np_image_size[1]))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            tile_counter += 1
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], np_image_size[1])
            y_max = min(y_min + tile_size[0], np_image_size[0])

            img = np_image[y_min:y_max, x_min:x_max, :]
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

            padded_img = pad_image(img, tile_size)

            if device == 'cpu':
                np_padded_img = padded_img.permute(0, 2, 3, 1).squeeze(0).numpy()
            else:
                np_padded_img = padded_img.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()

            trans_img = val_transforms(image=np_padded_img)['image'].to(device)
            trans_img = trans_img.unsqueeze(0)

            padded_logits = model(trans_img)
            padded_logits = torch.nn.functional.interpolate(padded_logits.unsqueeze(0),
                                                            size=(1, padded_img.shape[2], padded_img.shape[3]),
                                                            mode="nearest")[0]

            logits = padded_logits[:, :, :img.shape[2], :img.shape[3]]

            count_predictions[y_min:y_max, x_min:x_max] += 1
            if device == 'cpu':
                total_predictions[y_min:y_max, x_min:x_max] += logits.numpy().squeeze(0)[0]
            else:
                total_predictions[y_min:y_max, x_min:x_max] += logits.cpu().numpy().squeeze(0)[0]

    total_predictions /= count_predictions

    return total_predictions


def predict_ensemble(loaded_models_dict, image, threshold=0.5, device='cpu'):
    with torch.inference_mode():
        predictions = []
        for key, model in loaded_models_dict.items():
            if key.find('lp') > -1:
                pred_probs = predict_lp(model, image, device)
            else:
                pred_probs = predict_mmseg(model, image)
            predictions.append(pred_probs)
        average_prediction = np.mean(predictions, axis=0)
        final_segmentation = (average_prediction > threshold).astype(np.uint8)

    return final_segmentation


def main():
    input_root = '/input/images/adenocarcinoma-image'
    output_root = '/output/images/adenocarcinoma-mask'

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    config = 'config.py'
    checkpoint_lp = '/opt/ml/model/checkpoint_lp.pth'
    checkpoint_mmseg = '/opt/ml/model/checkpoint_mmseg.pth'
    models_use = {'dinov2lp': {'pretrained_encoder_name': 'facebook/dinov2-base',
                               'checkpoint': checkpoint_lp,
                               'work_dir': osp.join(os.getcwd(), 'ensemble_run'),
                               'device': 'cuda:0'
                               },
                  'dinov2_deeplabv3plus': {'config': config,
                                           'checkpoint': checkpoint_mmseg,
                                           'device': 'cuda:0'
                                           }
                  }

    loaded_models_dict = load_models(models_use)
    with torch.no_grad():
        for filename in os.listdir(input_root):
            if filename.endswith('.mha'):
                output_path = f'{output_root}/{filename}'
                try:
                    input_path = input_root + '/' + filename
                    image = read(input_path)
                    result = predict_ensemble(loaded_models_dict, image, threshold=0.57, device='cuda:0')
                    write(output_path, result)
                except Exception as error:
                    print(error)
