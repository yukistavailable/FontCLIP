import torch.nn as nn
import torchvision
from scipy.spatial import Delaunay
import torch
import numpy as np
from torch.nn import functional as nnf
from easydict import EasyDict
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from typing import List, Tuple
from PIL import Image 
from utils.transform_image import my_convert_to_rgb
from utils.tokenizer import tokenize

#  avoid circular import
# from optimize import Config

class FCLIPLoss(nn.Module):

    def generate_prompt(self, attribute):
        return f'{attribute} font'

    def __init__(self, cfg, clip_model, device, preprocess, clip_preprocess=None, image_init=None):
        super(FCLIPLoss, self).__init__()
        self.loss_weight = cfg.fclip_loss_w
        self.cfg = cfg

        self.clip_model = clip_model
        self.clip_model.eval()
        self.preprocess = preprocess
        self.image_init = image_init

        self.use_fclip_direction_loss = cfg.use_fclip_direction_loss
        self.use_fclip_direction_loss_vision = cfg.use_fclip_direction_loss_vision
        self.multiple_attributes = cfg.multiple_attributes
        self.multiple_text_encoders = cfg.multiple_text_encoders
        self.target_attributes = cfg.target_attributes
        self.target_attributes_weights = cfg.target_attributes_weights
        self.cos_sims = None
        self.visual_optimize = cfg.visual_optimize
        if self.visual_optimize:
            assert clip_preprocess is not None
        self.clip_preprocess = clip_preprocess

        assert self.multiple_attributes is False or self.multiple_text_encoders is False
        if self.use_fclip_direction_loss:
            assert not self.use_fclip_direction_loss_vision
            assert self.cfg.ref_semantic_concept is not None
        if self.use_fclip_direction_loss_vision:
            assert not self.use_fclip_direction_loss
            assert self.cfg.ref_image_file_path is not None
            assert self.cfg.image_file_path is not None

        if self.use_fclip_direction_loss or self.use_fclip_direction_loss_vision:
            assert image_init is not None
            image_init = self.clip_preprocess(image_init).unsqueeze(0).to(device)
            self.embedded_image_ref = self.clip_model.encode_image(image_init)

            if self.use_fclip_direction_loss:
                print(self.cfg.ref_semantic_concept)
                self.ref_tokenized_text = tokenize(self.cfg.ref_semantic_concept).to(device)
                ref_embedded_text = self.clip_model.encode_text(self.ref_tokenized_text)

                print(self.cfg.semantic_concept)
                self.tokenized_text = tokenize(self.cfg.semantic_concept).to(device)
                embedded_text = self.clip_model.encode_text(self.tokenized_text)
                self.delta_embedded_text = embedded_text - ref_embedded_text

                self.embedded_text = None
                for _ in range(self.cfg.batch_size):
                    if self.embedded_text is None:
                        self.embedded_text = embedded_text
                    else:
                        self.embedded_text = torch.cat((self.embedded_text, embedded_text), 0)
            else:
                self.ref_image = Image.open(self.cfg.ref_image_file_path)
                self.ref_image = my_convert_to_rgb(self.ref_image)
                self.ref_image = self.clip_preprocess(self.ref_image).unsqueeze(0).to(device)
                self.embedded_font_image_ref = self.clip_model.encode_image(self.ref_image)

                self.image_file_path = cfg.image_file_path
                image = Image.open(self.image_file_path)
                image = my_convert_to_rgb(image)
                image = self.clip_preprocess(image).unsqueeze(0).to(device)
                self.embedded_image = self.clip_model.encode_image(image)
                self.delta_embedded_image = self.embedded_image - self.embedded_font_image_ref



        elif self.visual_optimize:
            assert cfg.image_file_path is not None
            self.image_file_path = cfg.image_file_path
            image = Image.open(self.image_file_path)
            image = my_convert_to_rgb(image)
            image = self.clip_preprocess(image).unsqueeze(0).to(device)
            self.embedded_image = self.clip_model.encode_image(image)
        elif self.multiple_attributes or self.multiple_text_encoders:
            assert self.target_attributes is not None
            assert self.target_attributes_weights is not None
            assert len(self.target_attributes) == len(self.target_attributes_weights)

            self.embedded_texts = []
            for i in range(len(self.target_attributes)):
                embedded_text = None
                tmp_embedded_text = self.clip_model.encode_text(tokenize(self.generate_prompt(self.target_attributes[i])).to(device))
                for _ in range(self.cfg.batch_size):
                    if embedded_text is None:
                        embedded_text = tmp_embedded_text
                    else:
                        embedded_text = torch.cat((self.embedded_text, tmp_embedded_text), 0)
                self.embedded_texts.append(embedded_text)
        else:
            print(self.cfg.semantic_concept)
            self.tokenized_text = tokenize(self.cfg.semantic_concept).to(device)
            tmp_embedded_text = self.clip_model.encode_text(self.tokenized_text)
            self.embedded_text = None
            for _ in range(self.cfg.batch_size):
                if self.embedded_text is None:
                    self.embedded_text = tmp_embedded_text
                else:
                    self.embedded_text = torch.cat((self.embedded_text, tmp_embedded_text), 0)


    def get_scheduler(self, step=None):
        if step is not None:
            return self.loss_weight * np.exp(-(1/5)*((step-300)/(20)) ** 2)
        else:
            return self.loss_weight

    def forward(self, x, step=None):
        embedded_x = self.clip_model.encode_image(self.preprocess(x))

        if self.use_fclip_direction_loss:
            delta_embedded_x = embedded_x - self.embedded_image_ref
            fclip_direction_loss = torch.cosine_similarity(delta_embedded_x, self.delta_embedded_text, dim=-1)
            return - torch.cosine_similarity(embedded_x, self.embedded_text, dim=-1) * self.get_scheduler(step), - fclip_direction_loss
        
        if self.use_fclip_direction_loss_vision:
            delta_embedded_x = embedded_x - self.embedded_image_ref
            fclip_direction_loss = torch.cosine_similarity(delta_embedded_x, self.delta_embedded_image, dim=-1)
            return - torch.cosine_similarity(embedded_x, self.embedded_image, dim=-1) * self.get_scheduler(step), - fclip_direction_loss
            
        if self.visual_optimize:
            return - torch.cosine_similarity(embedded_x, self.embedded_image, dim=-1) * self.get_scheduler(step)

        if self.multiple_text_encoders:
            if self.cos_sims is None:
                self.cos_sims = []
                for i in range(len(self.target_attributes)):
                    cos_sim = torch.cosine_similarity(embedded_x, self.embedded_texts[i], dim=-1)
                    # remove gradient from cos_sim
                    cos_sim = cos_sim.detach()
                    self.cos_sims.append(cos_sim)

            loss = - torch.cosine_similarity(embedded_x, self.embedded_texts[0], dim=-1) * self.target_attributes_weights[0]
            for i in range(1, len(self.target_attributes)):
                tmp_cos_sim = torch.cosine_similarity(embedded_x, self.embedded_texts[i], dim=-1)
                tmp_loss = torch.abs(self.cos_sims[i] - tmp_cos_sim) * self.target_attributes_weights[i]
                loss += tmp_loss
            return loss

        if self.multiple_attributes:
            loss = 0
            for i in range(len(self.target_attributes)):
                tmp_loss = - torch.cosine_similarity(embedded_x, self.embedded_texts[i], dim=-1) * self.target_attributes_weights[i]
                loss += tmp_loss
            return loss
        

        return - torch.cosine_similarity(embedded_x, self.embedded_text, dim=-1) * self.get_scheduler(step)
        # return - torch.cosine_similarity(embedded_x, self.embedded_text, dim=-1)




class ToneLoss(nn.Module):
    def __init__(self, cfg):
        super(ToneLoss, self).__init__()
        #self.dist_loss_weight = cfg.loss.tone.dist_loss_weight
        self.dist_loss_weight = cfg.tone_loss_w
        self.im_init = None
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()
        self.blurrer = torchvision.transforms.GaussianBlur(kernel_size=(cfg.loss.tone.pixel_dist_kernel_blur,
                                                                        cfg.loss.tone.pixel_dist_kernel_blur), sigma=(cfg.loss.tone.pixel_dist_sigma))

    def set_image_init(self, im_init):
        self.im_init = im_init.permute(2, 0, 1).unsqueeze(0)
        self.init_blurred = self.blurrer(self.im_init)


    def get_scheduler(self, step=None):
        if step is not None:
            return self.dist_loss_weight * np.exp(-(1/5)*((step-300)/(20)) ** 2)
        else:
            return self.dist_loss_weight

    def forward(self, cur_raster, step=None):
        blurred_cur = self.blurrer(cur_raster)
        #return self.mse_loss(self.init_blurred.detach(), blurred_cur) * self.get_scheduler(step)
        return self.mse_loss(self.init_blurred.detach(), blurred_cur)
            

class ConformalLoss:
    def __init__(self, parameters: EasyDict, device: torch.device, target_letter: str, shape_groups):
        self.parameters = parameters
        self.device = device
        self.target_letter = target_letter
        self.shape_groups = shape_groups
        self.faces = self.init_faces(device)
        self.faces_roll_a = [torch.roll(self.faces[i], 1, 1) for i in range(len(self.faces))]

        with torch.no_grad():
            self.angles = []
            self.reset()


    def get_angles(self, points: torch.Tensor) -> torch.Tensor:
        points = points.to(self.device)
        angles_ = []
        for i in range(len(self.faces)):
            triangles = points[self.faces[i]]
            triangles_roll_a = points[self.faces_roll_a[i]]
            edges = triangles_roll_a - triangles
            length = edges.norm(dim=-1)
            edges = edges / (length + 1e-1)[:, :, None]
            edges_roll = torch.roll(edges, 1, 1)
            cosine = torch.einsum('ned,ned->ne', edges, edges_roll)
            angles = torch.arccos(cosine)
            angles_.append(angles)
        return angles_
    
    def get_letter_inds(self, letter_to_insert):
        for group, l in zip(self.shape_groups, self.target_letter):
            if l == letter_to_insert:
                letter_inds = group.shape_ids
                return letter_inds[0], letter_inds[-1], len(letter_inds)

    def reset(self):
        points = torch.cat([point.clone().detach() for point in self.parameters.point])
        self.angles = self.get_angles(points)

    def init_faces(self, device: torch.device) -> torch.tensor:
        faces_ = []
        for j, c in enumerate(self.target_letter):
            points_np = [self.parameters.point[i].clone().detach().cpu().numpy() for i in range(len(self.parameters.point))]
            start_ind, end_ind, shapes_per_letter = self.get_letter_inds(c)
            print(c, start_ind, end_ind)
            holes = []
            if shapes_per_letter > 1:
                holes = points_np[start_ind+1:end_ind]
            poly = Polygon(points_np[start_ind], holes=holes)
            poly = poly.buffer(0)
            points_np = np.concatenate(points_np)
            faces = Delaunay(points_np).simplices
            is_intersect = np.array([poly.contains(Point(points_np[face].mean(0))) for face in faces], dtype=np.bool)
            faces_.append(torch.from_numpy(faces[is_intersect]).to(device, dtype=torch.int64))
        return faces_

    def __call__(self) -> torch.Tensor:
        loss_angles = 0
        points = torch.cat(self.parameters.point)
        angles = self.get_angles(points)
        for i in range(len(self.faces)):
            loss_angles += (nnf.mse_loss(angles[i], self.angles[i]))
        return loss_angles



class LaplacianLoss:
    def __init__(self, parameters: EasyDict, device: torch.device, shape_groups, is_contour: bool=False, num_per_curve: int=10, skip_edge: bool=False, only_edge: bool=False, edge_cos_threshold: float=0.8):
        self.parameters = parameters
        self.is_contour = is_contour
        self.num_per_curve = num_per_curve
        self.skip_edge = skip_edge
        self.only_edge = only_edge
        if self.skip_edge:
            assert self.is_contour
            assert not self.only_edge
        if self.only_edge:
            assert not self.is_contour
            assert not self.skip_edge
        self.skip_mask = None
        assert len(shape_groups) == 1

        with torch.no_grad():
            self.laplacian_coordinates = []
            self.reset()

    def get_laplacian_coordinates(self, paths: List[torch.Tensor]) -> torch.Tensor:
        laplacian_coordinates_ = []
        if self.only_edge:
            for i in range(len(paths)):
                points = paths[i]
                assert (len(points) - 6) % 3 == 0 , f"Invalid path: len(path) must be 6 + 3*n (n >= 0) but len(path) = {len(points)}"
                num_curves = (len(points) - 6) // 3 + 2
                for i in range(num_curves):
                    start_point = points[i*3]
                    control_point_1 = points[i*3+1]
                    control_point_2 = points[i*3+2]
                    if i == num_curves - 1:
                        end_point = points[0]
                        control_point_3 = points[1]
                    else:
                        end_point = points[i*3+3]
                        control_point_3 = points[i*3+4]
                    if i == 0:
                        control_point_0 = points[-1]
                    else:
                        control_point_0 = points[i*3-1]

                    #TODO: resolve double count
                    laplacian_coordinates_.append(start_point - (control_point_0 + control_point_1) / 2)
                    laplacian_coordinates_.append(end_point - (control_point_2 + control_point_3) / 2)
        else:
            for i in range(len(paths)):
                points = paths[i]
                for j in range(len(points)):
                    if j == 0:
                        v1 = points[-1]
                    else:
                        v1 = points[j-1]
                    v2 = points[j]
                    if j == len(points) - 1:
                        v3 = points[0]
                    else:
                        v3 = points[j+1]
                    laplacian_coordinates_.append(v2 - (v1 + v3) / 2)
        laplacian_coordinates_ = torch.stack(laplacian_coordinates_)
        return laplacian_coordinates_
    

    def reset(self):
        paths = [point.clone().detach() for point in self.parameters.point]
        if self.is_contour:
            tmp_paths = []
            tmp_skip_mask = []
            for path in paths:
                points, skip_mask = sample_contour_from_bezier_curves(path, self.num_per_curve, skip_edge=self.skip_edge)
                tmp_paths.append(points)
                tmp_skip_mask.append(skip_mask)
            self.skip_mask = torch.stack(tmp_skip_mask)
            paths = tmp_paths

        self.laplacian_coordinates = self.get_laplacian_coordinates(paths)

    def __call__(self) -> torch.Tensor:
        if self.is_contour:
            paths = [sample_contour_from_bezier_curves(path, self.num_per_curve)[0] for path in self.parameters.point]
        else:
            paths = [path for path in self.parameters.point]
        laplacian_coordinates = self.get_laplacian_coordinates(paths)
        laplacian_coordinates_ = self.laplacian_coordinates
        if self.skip_edge:
            laplacian_coordinates = laplacian_coordinates[self.skip_mask.squeeze()]
            laplacian_coordinates_ = laplacian_coordinates_[self.skip_mask.squeeze()]

        diff_squared = (laplacian_coordinates - laplacian_coordinates_)**2
        diff_squared_sum = diff_squared.sum(dim=-1)
        loss_laplacian_coordinates = diff_squared_sum.mean()
        #loss_laplacian_coordinates = nnf.mse_loss(laplacian_coordinates, self.laplacian_coordinates)
        return loss_laplacian_coordinates

class LaplacianLossBetweenEdge:
    def __init__(self, parameters: EasyDict, threshold: float=-0.8):
        self.parameters = parameters
        self.threshold = threshold
        self.skip_mask = None

        with torch.no_grad():
            self.laplacian_coordinates = []
            self.reset()

    def get_cos(self, p1, p2, p3):
        return torch.cosine_similarity(p1 - p2, p3 - p2, dim=-1)

    def get_laplacian_coordinates(self, paths: List[torch.Tensor]) -> torch.Tensor:
        flag = False
        if self.skip_mask is None:
            flag = True
            self.skip_mask = []
        laplacian_coordinates_ = []
        for i in range(len(paths)):
            points = paths[i]
            assert (len(points) - 6) % 3 == 0 , f"Invalid path: len(path) must be 6 + 3*n (n >= 0) but len(path) = {len(points)}"
            num_curves = (len(points) - 6) // 3 + 2
            for i in range(num_curves):
                p2 = points[i*3]
                if i == 0:
                    p1 = points[-3]
                else:
                    p1 = points[i*3-3]
                if i == num_curves - 1:
                    p3 = points[0]
                else:
                    p3 = points[i*3+3]
                laplacian_coordinates_.append(p2 - (p1 + p3) / 2)
                cos = self.get_cos(p1, p2, p3)
                if flag:
                    self.skip_mask.append(cos > self.threshold)
        laplacian_coordinates_ = torch.stack(laplacian_coordinates_)
        if flag:
            self.skip_mask = torch.stack(self.skip_mask).bool()
        return laplacian_coordinates_
    

    def reset(self):
        paths = [point.clone().detach() for point in self.parameters.point]
        self.laplacian_coordinates = self.get_laplacian_coordinates(paths)

    def __call__(self) -> torch.Tensor:
        paths = [path for path in self.parameters.point]
        laplacian_coordinates = self.get_laplacian_coordinates(paths)[self.skip_mask]
        laplacian_coordinates_ = self.laplacian_coordinates[self.skip_mask]
        diff_squared = (laplacian_coordinates - laplacian_coordinates_)**2
        diff_squared_sum = diff_squared.sum(dim=-1)
        loss_laplacian_coordinates = diff_squared_sum.mean()
        #loss_laplacian_coordinates = nnf.mse_loss(laplacian_coordinates, self.laplacian_coordinates)
        return loss_laplacian_coordinates


class CosLoss:
    def __init__(self, parameters: EasyDict, device: torch.device, shape_groups, is_contour: bool=False, num_per_curve: int=10, 
                 skip_control_points: bool=False, skip_corners: bool=False, skip_corner_threshold: float=-0.9, skip_edge: bool=False, use_angle: bool=False):
        self.parameters = parameters
        self.is_contour = is_contour
        self.num_per_curve = num_per_curve
        self.skip_control_points = skip_control_points
        assert not (self.skip_control_points and self.is_contour)
        self.skip_corners = skip_corners
        self.skip_corner_threshold = skip_corner_threshold
        self.skip_edge = skip_edge
        self.use_angle = use_angle
        if self.skip_edge:
            assert self.is_contour

        # mask for skip corners
        # Be carefule that True means not skip
        self.skip_mask = None
        # mask for skip edge
        self.skip_edge_mask = None
        assert len(shape_groups) == 1

        with torch.no_grad():
            self.cos = []
            self.reset()


    def get_cos_or_angle(self, start_point, control_point_1, control_point_before_start):
        if self.use_angle:
            # calculate angle
            return torch.acos(torch.clamp(torch.cosine_similarity(control_point_1 - start_point, control_point_before_start - start_point, dim=-1), -1., 1.))
            # calculate tangent
            #return torch.atan2(control_point_1[1] - start_point[1], control_point_1[0] - start_point[0]) - torch.atan2(control_point_before_start[1] - start_point[1], control_point_before_start[0] - start_point[0])
        return torch.cosine_similarity(control_point_1 - start_point, control_point_before_start - start_point, dim=-1)


    def get_cos(self, paths: List[torch.Tensor]) -> torch.Tensor:
        cos_ = []
        if self.skip_control_points:
            skip_mask_flag = False
            if self.skip_corners and self.skip_mask is None:
                skip_mask_flag = True
                self.skip_mask = []
            for i in range(len(paths)):
                points = paths[i]

                # path should be a closed path which consists of only cubic bezier curves
                assert (len(points) - 6) % 3 == 0 , f"Invalid path: len(path) must be 6 + 3*n (n >= 0) but len(path) = {len(points)}"
                num_curves = (len(points) - 6) // 3 + 2
                for i in range(num_curves):
                    start_point = points[i*3]
                    control_point_1 = points[i*3+1]
                    if i == 0:
                        control_point_before_start = points[-1]
                    else:
                        control_point_before_start = points[i*3-1]
                    cos = self.get_cos_or_angle(start_point, control_point_1, control_point_before_start)
                    cos_.append(cos)
                    if self.skip_corners and skip_mask_flag:
                        if self.use_angle:
                            self.skip_mask.append(cos >= self.skip_corner_threshold)
                        else:
                            self.skip_mask.append(cos <= self.skip_corner_threshold)

        else:
            for i in range(len(paths)):
                points = paths[i]
                for j in range(len(points)):
                    if j == 0:
                        v1 = points[-1]
                    else:
                        v1 = points[j-1]
                    v2 = points[j]
                    if j == len(points) - 1:
                        v3 = points[0]
                    else:
                        v3 = points[j+1]
                    
                    cos = self.get_cos_or_angle(v2, v3, v1)
                    cos_.append(cos)

        cos_ = torch.stack(cos_)
        if self.skip_corners and skip_mask_flag:
            self.skip_mask = torch.stack(self.skip_mask)
        return cos_
    

    def reset(self):
        paths = [point.clone().detach() for point in self.parameters.point]
        if self.is_contour:
            tmp_paths = []
            tmp_skip_mask = []
            for path in paths:
                points, skip_mask = sample_contour_from_bezier_curves(path, self.num_per_curve, skip_edge=self.skip_edge)
                tmp_paths.append(points)
                tmp_skip_mask.extend(skip_mask)
            self.skip_edge_mask = torch.Tensor(tmp_skip_mask).bool()
            paths = tmp_paths

        self.cos = self.get_cos(paths)

    def __call__(self) -> torch.Tensor:
        if self.is_contour:
            paths = [sample_contour_from_bezier_curves(path, self.num_per_curve)[0] for path in self.parameters.point]
        else:
            paths = [path for path in self.parameters.point]
        cos = self.get_cos(paths)
        if self.skip_control_points and self.skip_corners:
            assert self.skip_mask is not None
            cos = cos[self.skip_mask]
            cos_ = self.cos[self.skip_mask]
            loss_cos = nnf.mse_loss(cos, cos_)
        elif self.is_contour:
            cos = cos[self.skip_edge_mask.squeeze()]
            cos_ = self.cos[self.skip_edge_mask.squeeze()]
            loss_cos = nnf.mse_loss(cos, cos_)
        else:
            loss_cos = nnf.mse_loss(cos, self.cos)
        return loss_cos

class XingLoss(nn.Module):
    def __init__(self, parameters: EasyDict):
        super(XingLoss, self).__init__()
        self.parameters = parameters
        self.relu = nn.ReLU()

    def get_outer_product(self, v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    def get_sign_of_outer_product(self, v1, v2):
        outer_product = self.get_outer_product(v1, v2)
        if outer_product > 0:
            return 1
        else:
            return 0

    def get_xing(self, paths: List[torch.Tensor]) -> torch.Tensor:
        xings_ = []
        for i in range(len(paths)):
            points = paths[i]
            assert (len(points) - 6) % 3 == 0 , f"Invalid path: len(path) must be 6 + 3*n (n >= 0) but len(path) = {len(points)}"
            num_curves = (len(points) - 6) // 3 + 2
            for i in range(num_curves):
                p1 = points[i*3]
                p2 = points[i*3+1]
                p3 = points[i*3+2]
                if i == num_curves - 1:
                    p4 = points[0]
                else:
                    p4 = points[i*3+3]
                D1 = self.get_sign_of_outer_product(p2 - p1, p3 - p2)
                D2 = (self.get_outer_product(p2 - p1, p4 - p3)) / (torch.norm(p2 - p1) * torch.norm(p4 - p3))
                tmp_xing = D1 * self.relu(-D2) + (1 - D1) * self.relu(D2)
                xings_.append(tmp_xing)
        xings_ = torch.stack(xings_)
        return xings_

    def __call__(self) -> torch.Tensor:
        paths = [path for path in self.parameters.point]
        xings = self.get_xing(paths)
        # convert xings to loss
        loss_xing = torch.mean(xings)
        return loss_xing

class DirectionLoss(nn.Module):
    def __init__(self, parameters: EasyDict):
        super(DirectionLoss, self).__init__()
        self.parameters = parameters
        with torch.no_grad():
            self.vs = self.get_vectors(self.parameters.point)

    def get_vectors(self, paths: List[torch.Tensor]) -> torch.Tensor:
        vs_ = []
        for i in range(len(paths)):
            points = paths[i]
            assert (len(points) - 6) % 3 == 0 , f"Invalid path: len(path) must be 6 + 3*n (n >= 0) but len(path) = {len(points)}"
            num_curves = (len(points) - 6) // 3 + 2
            for i in range(num_curves):
                p1 = points[i*3]
                #p2 = points[i*3+1]
                #p3 = points[i*3+2]
                if i == num_curves - 1:
                    p4 = points[0]
                else:
                    p4 = points[i*3+3]
                v = p4 - p1
                vs_.append(v)
        vs_ = torch.stack(vs_)
        return vs_

    def __call__(self) -> torch.Tensor:
        paths = [path for path in self.parameters.point]
        vs = self.get_vectors(paths)
        # calculate cos
        cos = torch.sum(vs * self.vs, dim=1) / (torch.norm(vs, dim=1) * torch.norm(self.vs, dim=1))
        return torch.Tensor(1 - torch.mean(cos))


def sample_contour_from_bezier_curves(path: torch.Tensor, num_per_curve: int=10, skip_edge: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters
    ----------
    path : torch.Tensor
        shape: (6 + 3*n, 2)
    num_per_curve : int, optional
        The number of points sampled from each bezier curve, by default 10
    
    Returns
    -------
    torch.Tensor
        shape: (num_per_curve * n, 2)
    """

    contour = []
    # len(path) must be 6 + 3*n (n >= 0) since path consists of only bezier curves
    assert (len(path) - 6) % 3 == 0 , f"Invalid path: len(path) must be 6 + 3*n (n >= 0) but len(path) = {len(path)}"
    num_curves = (len(path) - 6) // 3 + 2
    skip_mask = []
    for i in range(num_curves):
        start_point = path[i*3]
        control_point_1 = path[i*3+1]
        control_point_2 = path[i*3+2]
        end_point = path[0]
        if i != num_curves - 1:
            end_point = path[i*3+3]
        for t in np.linspace(0, 1, num_per_curve):
            if t != 1.:
                contour.append((1-t)**3 * start_point + 3*(1-t)**2 * t * control_point_1 + 3*(1-t) * t**2 * control_point_2 + t**3 * end_point)
                if skip_edge:
                    if t != 0.:
                        skip_mask.append(True)
                    else:
                        skip_mask.append(False)
                else:
                    skip_mask.append(True)
            
    assert len(contour) == len(skip_mask)
    contour = torch.stack(contour)
    skip_mask = torch.tensor(skip_mask)
    return (contour, skip_mask)


class L2Loss(nn.Module):
    def __init__(self, cfg) -> None:
        super(L2Loss, self).__init__()
        self.cfg = cfg
        self.x: torch.Tensor = cfg.target_img

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean((x - self.x)**2)