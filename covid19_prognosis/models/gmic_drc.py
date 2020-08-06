import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from covid19_prognosis.models import resnet
from torch.autograd import Variable
import os


def generate_mask_uplft(input_image, window_shape, upper_left_points, gpu_number):
    """
    Function that generates mask that sets crops given upper_left
    corners to 0
    :param input_image:
    :param window_shape:
    :param upper_left_points:
    """
    N, C, H, W = input_image.size()
    window_h, window_w = window_shape
    # get the positions of masks
    mask_x_min = upper_left_points[:,:,0]
    mask_x_max = upper_left_points[:,:,0] + window_h
    mask_y_min = upper_left_points[:,:,1]
    mask_y_max = upper_left_points[:,:,1] + window_w
    # generate masks
    mask_x = Variable(torch.arange(0, H).view(-1, 1).repeat(N, C, 1, W))
    mask_y = Variable(torch.arange(0, W).view(1, -1).repeat(N, C, H, 1))
    if gpu_number is not None:
        device = torch.device("cuda:{}".format(gpu_number))
        mask_x = mask_x.cuda().to(device)
        mask_y = mask_y.cuda().to(device)
    x_gt_min = mask_x.float() >= mask_x_min.unsqueeze(-1).unsqueeze(-1).float()
    x_ls_max = mask_x.float() < mask_x_max.unsqueeze(-1).unsqueeze(-1).float()
    y_gt_min = mask_y.float() >= mask_y_min.unsqueeze(-1).unsqueeze(-1).float()
    y_ls_max = mask_y.float() < mask_y_max.unsqueeze(-1).unsqueeze(-1).float()

    # since logic operation is not supported for variable
    # I used * for logic AND
    selected_x = x_gt_min * x_ls_max
    selected_y = y_gt_min * y_ls_max
    selected = selected_x * selected_y
    mask = 1 - selected.float()
    return mask

def get_max_window(input_image, window_shape, pooling_logic="avg"):
    """
    Function that makes a sliding window of size window_shape over the
    input_image and return the UPPER_LEFT corner index with max sum
    :param input_image: N*C*H*W
    :param window_shape: h*w
    :return: N*C*2 tensor
    """
    N, C, H, W = input_image.size()
    if pooling_logic == "avg":
        # use average pooling to locate the window sums
        pool_map = torch.nn.functional.avg_pool2d(input_image, window_shape, stride=1)
    elif pooling_logic in ["std", "avg_entropy"]:
        # create sliding windows
        output_size = (H - window_shape[0] + 1, W - window_shape[1] + 1)
        sliding_windows = F.unfold(input_image, kernel_size=window_shape).view(N,C, window_shape[0]*window_shape[1], -1)
        # apply aggregation function on each sliding windows
        if pooling_logic == "std":
            agg_res = sliding_windows.std(dim=2, keepdim=False)
        elif pooling_logic == "avg_entropy":
            agg_res = -sliding_windows*torch.log(sliding_windows)-(1-sliding_windows)*torch.log(1-sliding_windows)
            agg_res = agg_res.mean(dim=2, keepdim=False)
        # merge back
        pool_map = F.fold(agg_res, kernel_size=(1, 1), output_size=output_size)
    _, _, _, W_map = pool_map.size()
    # transform to linear and get the index of the max val locations
    _, max_linear_idx = torch.max(pool_map.view(N, C, -1), -1)
    # convert back to 2d index
    max_idx_x = max_linear_idx / W_map
    max_idx_y = max_linear_idx - max_idx_x * W_map
    # put together the 2d index
    upper_left_points = torch.cat([max_idx_x.unsqueeze(-1), max_idx_y.unsqueeze(-1)], dim=-1)
    return upper_left_points

def top_k_percent_pooling(saliency_map, percent_r):
    """
    Function that perform the top k percent pooling
    :param saliency_map:
    :param percent_r:
    :return:
    """
    N, C, H, W = saliency_map.size()
    cam_flatten = saliency_map.view(N, C, -1)
    top_k = int(round(H * W * percent_r))
    selected_area = cam_flatten.topk(top_k, dim=2)[0]
    saliency_map_pred = selected_area.mean(dim=2)
    return saliency_map_pred

def make_sure_in_range(val, min_val, max_val):
    """
    Function that make sure that min < val < max; otherwise return the limit value
    """
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val

def crop_pytorch(original_img_pytorch, crop_shape, crop_position, out,
                 method="center", background_val="min"):
    """
    Function that take a crop on the original image.
    Use PyTorch to do this.
    :param original_img_pytorch: (N,C,H,W) PyTorch Tensor
    :param crop_shape: (h, w) integer tuple
    :param method: supported in ["center", "upper_left"]
    :return: (N, K, h, w) PyTorch Tensor
    """
    # retrieve inputs
    _, H, W = original_img_pytorch.shape
    crop_x, crop_y = crop_position
    x_delta, y_delta = crop_shape

    # locate the four corners
    if method == "center":
        min_x = int(np.round(crop_x - x_delta / 2))
        max_x = int(np.round(crop_x + x_delta / 2))
        min_y = int(np.round(crop_y - y_delta / 2))
        max_y = int(np.round(crop_y + y_delta / 2))
    elif method == "upper_left":
        min_x = int(np.round(crop_x))
        max_x = int(np.round(crop_x + x_delta))
        min_y = int(np.round(crop_y))
        max_y = int(np.round(crop_y + y_delta))

    # make sure that the crops are in range
    min_x = make_sure_in_range(min_x, 0, H)
    max_x = make_sure_in_range(max_x, 0, H)
    min_y = make_sure_in_range(min_y, 0, W)
    max_y = make_sure_in_range(max_y, 0, W)

    # somehow background is normalized to this number
    if background_val == "min":
        out[:,:,:] = original_img_pytorch.min()
    else:
        out[:, :, :] = background_val
    real_x_delta = max_x - min_x
    real_y_delta = max_y - min_y
    origin_x = crop_shape[0] - real_x_delta
    origin_y = crop_shape[1] - real_y_delta
    out[:, origin_x:, origin_y:] = original_img_pytorch[:, min_x:max_x, min_y:max_y]


def top_k_percent_pooling(saliency_map, percent_r):
    """
    Function that perform the top k percent pooling
    :param saliency_map:
    :param percent_r:
    :return:
    """
    N, C, W, H = saliency_map.size()
    cam_flatten = saliency_map.view(N, C, -1)
    top_k = int(round(W * H * percent_r))
    selected_area = cam_flatten.topk(top_k, dim=2)[0]
    saliency_map_pred = selected_area.mean(dim=2)
    return saliency_map_pred

def build_feature_extractor(key, pretrained_path):
    """
    Function that retrieves the feature extractor network
    :param key:
    :param pretrained_path:
    :return:
    """
    # resnet
    if key == "resnet18":
        if pretrained_path is None:
            model = resnet.resnet18(pretrained = False)
        else:
            model = resnet.ResNet(block=resnet.BasicBlock, layers=[2, 2, 2, 2])
        feature_dim = 512
    elif key == "resnet34":
        if pretrained_path is None:
            model = resnet.resnet34(pretrained = False)
        else:
            model = resnet.ResNet(block=resnet.BasicBlock, layers=[3, 4, 6, 3])
        feature_dim = 512
    elif key == "resnet50":
        model = resnet.ResNet(block=resnet.Bottleneck, layers=[3, 4, 6, 3])
        feature_dim = 2048
    elif key == "resnet101":
        model = resnet.ResNet(block=resnet.Bottleneck, layers=[3, 4, 23, 3])
        feature_dim = 2048
    elif key == "resnet152":
        model = resnet.ResNet(block=resnet.Bottleneck, layers=[3, 8, 36, 3])
        feature_dim = 2048
    # DenseNet family
    elif key in ["densenet121_nih", "densenet121", "densenet121_langone"]:
        model = densenet.densenet121()
        feature_dim = 1024
    elif key == "densenet161":
        model = densenet.densenet161()
        feature_dim = 2208
    elif key == "densenet169":
        model = densenet.densenet169()
        feature_dim = 1664
    elif key == "densenet201":
        model = densenet.densenet201()
        feature_dim = 1920
    # EfficientNet family
    elif key == 'efficientnet':
        model =  EfficientNet.from_pretrained('efficientnet-b4', num_classes=9)
        feature_dim = 1792
    elif key == 'efficientnet2':
        model =  EfficientNet2.from_pretrained('efficientnet-b4', num_classes=9)
        feature_dim = 1792
    else:
        raise ValueError("Unrecognized feature extractor {0}".format(key))
    if pretrained_path == 'None':
        pretrained_path = None
    # load pretrained weights
    if pretrained_path is not None:
        if '.pt' in pretrained_path or '.pth' in pretrained_path:
            pretrain_dir = pretrained_path
        else:
            pretrain_dir = os.path.join(pretrained_path, "{0}.pth".format(key))
        if "nih" in key or "langone" in key:
            state_dict = torch.load(pretrain_dir)
            for key in list(state_dict.keys()):
                new_key = key.replace("densenet121.", "").replace("norm.", "norm").replace("conv.", "conv").replace("normweight", "norm.weight").replace("normbias", "norm.bias").replace("normrunning_mean", "norm.running_mean").replace("normrunning_var", "norm.running_var").replace("convweight", "conv.weight")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
            model.load_state_dict(state_dict, strict=False)
        else:
            state_dict = torch.load(pretrain_dir)
            for key in list(state_dict.keys()):
                if "feature_extractor" in key:
                    new_key = key.replace("feature_extractor.", "")
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            model.load_state_dict(state_dict, strict=False)
        print("pretrained weights loaded from {0}".format(pretrain_dir))
    return model, feature_dim


class BaselineModel(nn.Module):
    """
    Baseline model used for pre-training
    """
    def __init__(self, parameters):
        super().__init__()
        self.experiment_parameters = parameters
        self.feature_extractor, feature_dim = build_feature_extractor(parameters["feature_extractor"],
                                                         parameters["pretrained_extractor_path"])
        self.classifier = nn.Linear(feature_dim, parameters["number_classes"], bias=False)

    def forward(self, imgs):
        feature_at_all_levels = self.feature_extractor(imgs)
        x5 = feature_at_all_levels[-1]
        return torch.sigmoid(self.classifier(x5))


class RetrieveROIModule():
    """
    A Regional Proposal Network instance that computes the locations of the crops
    Greedy select crops with largest sums
    """
    def __init__(self, parameters):
        self.crop_method = "upper_left"
        self.num_crops_per_class = parameters["K"]
        self.crop_shape = parameters["crop_shape"]
        self.gpu_number = None if parameters["device_type"]!="gpu" else parameters["gpu_number"]

    def forward(self, x_original, cam_size, h_small):
        """
        Function that use the low-res image to determine the position of the high-res crops
        :param x_original: N, C, H, W pytorch tensor
        :param cam_size: (h, w)
        :param h_small: N, C, h_h, w_h pytorch tensor
        :return: N, num_classes*k, 2 numpy matrix; returned coordinates are corresponding to x_small
        """
        # retrieve parameters
        _, _, H, W = x_original.size()
        (h, w) = cam_size
        N, C, h_h, w_h = h_small.size()

        # make sure that the size of h_small == size of cam_size
        assert h_h == h, "h_h({})!=h({})".format(h_h,h)
        assert w_h == w, "w_h({})!=w({})".format(w_h,w)

        # adjust crop_shape since crop shape is based on the original image
        crop_x_adjusted = int(np.round(self.crop_shape[0] * h / H))
        crop_y_adjusted = int(np.round(self.crop_shape[1] * w / W))
        crop_shape_adjusted = (crop_x_adjusted, crop_y_adjusted)

        # greedily find the box with max sum of weights
        current_images = h_small
        all_max_position = []
        # combine channels
        max_vals = current_images.view(N, C, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        min_vals = current_images.view(N, C, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        range_vals = max_vals - min_vals
        normalize_images = current_images - min_vals
        normalize_images = normalize_images / range_vals
        current_images = normalize_images.sum(dim=1, keepdim=True)

        for _ in range(self.num_crops_per_class):
            max_pos = get_max_window(current_images, crop_shape_adjusted, "avg")
            all_max_position.append(max_pos)
            mask = generate_mask_uplft(current_images, crop_shape_adjusted, max_pos, self.gpu_number)
            current_images = current_images * mask
        return torch.cat(all_max_position, dim=1).data.cpu().numpy()

class AttentionModule(nn.Module):
    """
    The attention module takes multiple hidden representations and compute the attention-weighted average
    Use Gated Attention Mechanism in https://arxiv.org/pdf/1802.04712.pdf
    """
    def __init__(self, in_dim, out_class=1, attn_dim=128):
        super(AttentionModule, self).__init__()
        # The gated attention mechanism
        self.mil_attn_V = nn.Linear(in_dim, attn_dim, bias=False)
        self.mil_attn_U = nn.Linear(in_dim, attn_dim, bias=False)
        self.mil_attn_w = nn.Linear(attn_dim, 1, bias=False)
        # classifier
        self.classifier_linear = nn.Linear(in_dim, out_class, bias=False)

    def forward(self, h_crops):
        """
        Function that takes in the hidden representations of crops and use attention to generate a single hidden vector
        :param h_small:
        :param h_crops:
        :return:
        """
        batch_size, num_crops, h_dim = h_crops.size()
        h_crops_reshape = h_crops.view(batch_size * num_crops, h_dim)
        # calculate the attn score
        attn_projection = torch.sigmoid(self.mil_attn_U(h_crops_reshape)) * \
                          torch.tanh(self.mil_attn_V(h_crops_reshape))
        attn_score = self.mil_attn_w(attn_projection)
        # use softmax to map score to attention
        attn_score_reshape = attn_score.view(batch_size, num_crops)
        attn = F.softmax(attn_score_reshape, dim=1)

        # final hidden vector
        z_weighted_avg = torch.sum(attn.unsqueeze(-1) * h_crops, 1)

        # map to the final layer
        y_crops = torch.sigmoid(self.classifier_linear(z_weighted_avg))
        return z_weighted_avg, attn, y_crops


class GMIC(nn.Module):
    def __init__(self, parameters):
        super(GMIC, self).__init__()

        # save parameters
        self.experiment_parameters = parameters
        self.cam_size = parameters["cam_size"]
        parameters["global_only"] = False
        self.global_only = parameters["global_only"]

        # construct networks
        # global network
        self.global_network, feature_dim = build_feature_extractor(parameters["feature_extractor"],
                                                                      parameters["pretrained_extractor_path"])
        self.sm_transformer = nn.Conv2d(feature_dim, parameters["number_classes"], kernel_size=1, stride=1, bias=False)
        self.percent_r = self.experiment_parameters["percent_r"]

        # detection module
        self.retrieve_roi_crops = RetrieveROIModule(self.experiment_parameters)

        # detection network
        self.local_network, patch_feature_dim = build_feature_extractor(parameters["local_network"],
                                                    parameters["pretrained_local_network_path"])

        # # MIL module
        self.attention_module = AttentionModule(patch_feature_dim, out_class=parameters["number_classes"])

        # # fusion branch
        self.fusion_dnn = nn.Linear(feature_dim+patch_feature_dim, parameters["number_classes"], bias=False)
        


    def _convert_crop_position(self, crops_x_small, cam_size, x_original):
        """
        Function that converts the crop locations from cam_size to x_original
        :param crops_x_small: N, k*c, 2 numpy matrix
        :param cam_size: (h,w)
        :param x_original: N, C, H, W pytorch variable
        :return: N, k*c, 2 numpy matrix
        """
        # retrieve the dimension of both the original image and the small version
        h, w = cam_size
        _, _, H, W = x_original.size()

        # interpolate the 2d index in h_small to index in x_original
        top_k_prop_x = crops_x_small[:, :, 0] / h
        top_k_prop_y = crops_x_small[:, :, 1] / w
        # sanity check
        assert np.max(top_k_prop_x) <= 1.0, "top_k_prop_x >= 1.0"
        assert np.min(top_k_prop_x) >= 0.0, "top_k_prop_x <= 0.0"
        assert np.max(top_k_prop_y) <= 1.0, "top_k_prop_y >= 1.0"
        assert np.min(top_k_prop_y) >= 0.0, "top_k_prop_y <= 0.0"
        # interpolate the crop position from cam_size to x_original
        top_k_interpolate_x = np.expand_dims(np.around(top_k_prop_x * H), -1)
        top_k_interpolate_y = np.expand_dims(np.around(top_k_prop_y * W), -1)
        top_k_interpolate_2d = np.concatenate([top_k_interpolate_x, top_k_interpolate_y], axis=-1)
        return top_k_interpolate_2d

    def _retrieve_crop(self, x_original_pytorch, crop_positions, crop_method):
        """
        Function that takes in the original image and cropping position and returns the crops
        :param x_original_pytorch: PyTorch Tensor array (N,C,H,W)
        :param crop_positions:
        :return:
        """
        batch_size, num_crops, _ = crop_positions.shape
        _, C, _, _ = x_original_pytorch.size()
        crop_h, crop_w = self.experiment_parameters["crop_shape"]

        output = torch.ones((batch_size, num_crops, C, crop_h, crop_w))
        if self.experiment_parameters["device_type"] == "gpu":
            device = torch.device("cuda:{}".format(self.experiment_parameters["gpu_number"]))
            output = output.cuda().to(device)
        for i in range(batch_size):
            for j in range(num_crops):
                crop_pytorch(x_original_pytorch[i, :, :, :],
                                                    self.experiment_parameters["crop_shape"],
                                                    crop_positions[i,j,:],
                                                    output[i,j,:,:,:],
                                                    method=crop_method)
        return output

    def calculate_accuracy_loss(self, criterion, input_label_variable):
        """
        Customize the loss calculation for all level
        :param criterion:
        :param input_label_variable:
        :return:
        """
        loss = criterion(self.y_global, input_label_variable)
        if "y_local" in dir(self):
            loss += criterion(self.y_local, input_label_variable)
        if "y_fusion" in dir(self):
            loss += criterion(self.y_fusion, input_label_variable)
        return loss


    def forward(self, x_original):
        """
        :param x_original: N,H,W,C numpy matrix
        """
        # global network: x_small -> class activation map
        feature_at_all_levels = self.global_network(x_original)
        if type(feature_at_all_levels) == list:
            h_g = feature_at_all_levels[-2]
        else:
            h_g = feature_at_all_levels
        self.saliency_map = torch.sigmoid(self.sm_transformer(h_g))

        # calculate y_global
        # note that y_global is not directly used in inference
        self.y_global = top_k_percent_pooling(self.saliency_map, self.percent_r)
        if self.global_only:
            return self.y_global

        # region proposal network
        small_x_locations = self.retrieve_roi_crops.forward(x_original, self.cam_size, self.saliency_map)

        # convert crop locations that is on self.cam_size to x_original
        self.patch_locations = self._convert_crop_position(small_x_locations, self.cam_size, x_original)

        # patch retriever
        crops_variable = self._retrieve_crop(x_original, self.patch_locations, self.retrieve_roi_crops.crop_method)
        self.patches = crops_variable.data.cpu().numpy()

        # local network
        batch_size, num_crops, C, I, J = crops_variable.size()
        crops_variable = crops_variable.view(batch_size * num_crops, C, I, J)
        patch_features = self.local_network.forward(crops_variable)
        if type(patch_features) == list:
            h_crops = patch_features[-1].view(batch_size, num_crops, -1)
        else:
            h_crops = patch_features.view(batch_size, num_crops, -1)

        # MIL module
        # y_local is not directly used during inference
        z, self.patch_attns, self.y_local = self.attention_module.forward(h_crops)

        # fusion branch
        # use max pooling to collapse the feature map
        g1, _ = torch.max(h_g, dim=2)
        global_vec, _ = torch.max(g1, dim=2)

        
        concat_vec = torch.cat([global_vec, z], dim=1)
        self.y_fusion = torch.sigmoid(self.fusion_dnn(concat_vec))
        return self.y_fusion
