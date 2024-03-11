from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from misc_functions import get_example_params, save_class_activation_images
from scipy import spatial

import random
class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, enc, rel, target_layer):
        self.enc = enc
        self.rel = rel
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.enc._modules.items():
            x = module(x)  
            if module_pos == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x1, x2):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output1, x1 = self.forward_pass_on_convolutions(x1)
        conv_output2, x2 = self.forward_pass_on_convolutions(x2)
        x = self.rel(torch.cat((x1.squeeze(), x2.squeeze())).reshape(128, 19, 19).unsqueeze(0))
        return conv_output1, conv_output2, x


class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, enc, rel, target_layer):
        self.enc = enc
        self.rel = rel
        self.enc.eval()
        self.rel.eval()
        # Define extractor
        self.extractor = CamExtractor(self.enc, self.rel, target_layer)

    def generate_cam(self, input_image1, input_image2):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output1, conv_output2, model_output = \
            self.extractor.forward_pass(input_image1, input_image2)
        
        conv_outputs = [conv_output1, conv_output2]
        input_images = [input_image1, input_image2]
        cams = []
        for img_i, (conv_output, input_image) in enumerate(zip(conv_outputs, input_images)):
            # Get convolution outputs
            target = conv_outputs[img_i][0]
            target2 = conv_outputs[1 - img_i][0]
            # Create empty numpy array for cam
            cam = np.ones(target.shape[1:], dtype=np.float32)
            # Multiply each weight with its conv output and then, sum
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
                w = F.softmax(self.extractor.forward_pass(input_images[0]*\
                    norm_saliency_map1, input_images[1]*\
                    norm_saliency_map2)[2],dim=1)[0][0]
                cam += w.detach().data.numpy() * target[i, :, :].detach().data.numpy()
            cam = np.maximum(cam, 0)
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
            cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                        input_image.shape[3]), Image.ANTIALIAS))/255
            cams.append(cam)
        return cams[0], cams[1]


if __name__ == '__main__':
    # Get params
    target_example1 = 0
    target_example2 = 1
    (original_image1, original_image2, prep_img1, prep_img2, target_class, file_name_to_export1,\
         file_name_to_export2, pretrained_enc, rel) =\
        get_example_params(target_example1, target_example2)
    # Score cam
    score_cam = ScoreCam(pretrained_enc, rel, target_layer='layer4')
    # Generate cam mask
    cam1, cam2 = score_cam.generate_cam(prep_img1, prep_img2)
    # Save mask
    save_class_activation_images(original_image1, cam1, file_name_to_export1)
    save_class_activation_images(original_image2, cam2, file_name_to_export2)
    print('Score cam completed')