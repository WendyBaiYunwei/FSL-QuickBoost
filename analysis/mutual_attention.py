from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from misc_functions import get_example_params, save_class_activation_images
from scipy import spatial
from scipy.ndimage import zoom
import random
class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, enc, target_layer):
        self.enc = enc
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        modules = list(self.enc._modules.items())
        target_layer = modules[-3][0]
        emb_layer = modules[-1][0]
        conv_output = None
        emb = None
        for module_pos, module in self.enc._modules.items():
            x = module(x)  
            # if module_pos == self.target_layer:
            if module_pos == target_layer:
                conv_output = x  # Save the convolution output on that layer
            if module_pos == emb_layer:
                emb = x
        return conv_output, x, emb

    def forward_pass(self, x1, x2):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        
        conv_output1, x1, emb1 = self.forward_pass_on_convolutions(x1)
        conv_output2, x2, emb2 = self.forward_pass_on_convolutions(x2)
        emb1 = emb1.squeeze()
        emb2 = emb2.squeeze()
        rel = torch.dot(emb1, emb2) / (torch.norm(emb1, p=2) * torch.norm(emb2, p=2) + 1e-8)
        return conv_output1, conv_output2, rel


class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, enc, target_layer):
        self.enc = enc
        self.enc.eval()
        # Define extractor
        self.extractor = CamExtractor(self.enc, target_layer)

    def upscale(self, cam, input_image):
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                    input_image.shape[3]), Image.ANTIALIAS))/255
        # print(input_image.shape)
        # exit()
        # cam = zoom(cam, zoom=(input_image.shape[2] / len(cam), input_image.shape[3] / len(cam)), order=1)
        return cam

    def generate_cam(self, input_image1, input_image2):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output1, conv_output2, rel = \
            self.extractor.forward_pass(input_image1, input_image2)

        conv_outputs = [conv_output1, conv_output2]
        input_images = [input_image1, input_image2]
        cams = []
        # Get convolution outputs
        target = conv_outputs[0][0]
        target2 = conv_outputs[1][0]
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        cam2 = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        weight_sum = 0
        for i in range(len(target)):
            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :],0),0)
            # Upsampling to input size
            saliency_map = F.interpolate(saliency_map, size=(84, 84), mode='bilinear', align_corners=False)

            saliency_map2 = torch.unsqueeze(torch.unsqueeze(target2[i, :, :],0),0)
            # Upsampling to input size
            saliency_map2 = F.interpolate(saliency_map2, size=(84, 84), mode='bilinear', align_corners=False)
            if saliency_map.max() == saliency_map.min() or saliency_map2.max() == saliency_map2.min():
                continue
            # Scale between 0-1
            norm_saliency_map1 = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            norm_saliency_map2 = (saliency_map2 - saliency_map2.min()) / (saliency_map2.max() - saliency_map2.min())
            # Mutual attention weight
            w = self.extractor.forward_pass(input_image1*\
                norm_saliency_map1, input_image2*\
                norm_saliency_map2)[2].item()
            # weight_sum += w
            cam += w * target[i, :, :].detach().data.numpy()
            cam2 += w * target2[i, :, :].detach().data.numpy()
            # if i == 10:
            #     break
            # cam /= weight_sum
        cam = self.upscale(cam, input_image1)
        cam2 = self.upscale(cam2, input_image2)
            
        return cam, cam2


if __name__ == '__main__':
    # Get params
    encoder_type = 'Res12'
    img_id = encoder_type + '-7-2'# trial id-qrySptPair id
    target_example1 = 0
    target_example2 = 1
    (original_image1, original_image2, prep_img1, prep_img2, target_class, file_name_to_export1,\
         file_name_to_export2, pretrained_enc) =\
        get_example_params(target_example1, target_example2, encoder_type)
    # Score cam
    score_cam = ScoreCam(pretrained_enc, target_layer='layer4')
    # Generate cam mask
    cam1, cam2 = score_cam.generate_cam(prep_img1, prep_img2)
    # Save mask
    save_class_activation_images(original_image1, cam1, file_name_to_export1, img_id)
    save_class_activation_images(original_image2, cam2, file_name_to_export2, img_id)
    print('Score cam completed')