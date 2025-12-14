from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union, OrderedDict
import json
import numpy as np
import pyquaternion as pyquat
from PIL import Image
import glob

from nerfstudio.process_data import colmap_utils, hloc_utils, process_data_utils
from nerfstudio.process_data.base_converter_to_nerfstudio_dataset import BaseConverterToNerfstudioDataset
from nerfstudio.process_data.colmap_converter_to_nerfstudio_dataset import ColmapConverterToNerfstudioDataset
from nerfstudio.process_data.process_data_utils import CAMERA_MODELS
from nerfstudio.utils import install_checks
from nerfstudio.utils.rich_utils import CONSOLE, status

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import os, sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from dataclasses import dataclass, field



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CenterNet2
sys.path.append(os.path.join(BASE_DIR, "segmentation/Detic/third_party/CenterNet2"))

# Detic
sys.path.append(os.path.join(BASE_DIR, "segmentation/Detic"))

# GroundingDINO
sys.path.append(os.path.join(BASE_DIR, "segmentation/groundedSAM/GroundingDINO"))

from centernet.config import add_centernet_config




from detic.config import add_detic_config
import PIL
import imageio
from detic.predictor import VisualizationDemo



import supervision as sv
from cf_nerf.segmentation.groundedSAM.GroundingDINO.groundingdino.util.inference import Model
# from segment_anything import sam_model_registry, SamPredictor
from segment_anything_hq import sam_model_registry, SamPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.functional import dice


# DEVICE = torch.device('cpu')


class BaseImageSegmentation:
    def __init__(self, device, debug):
        self.device = device
        self.debug = debug

        self.iou_metric = BinaryJaccardIndex()

    def run(self, **kwargs):
        pass

    def validate(self, validation_path: Path, output_filename: Path, pred_mask: np.ndarray, debug=False):
        gt_mask = validation_path / output_filename.name
        gt_mask_image = cv2.imread(gt_mask.__str__(), cv2.IMREAD_UNCHANGED)

        gt_mask_image = gt_mask_image.astype(np.int16)
        mask = np.asarray(pred_mask)
        mask = mask.astype(np.int16)

        gt_mask_image[gt_mask_image > 0] = 1
        mask[mask > 0] = 1

        iou_val = self.iou_metric(torch.asarray(mask), torch.asarray(gt_mask_image))
        dice_score = dice(torch.asarray(mask), torch.asarray(gt_mask_image))

        print("IOU: {}, DICE: {}".format(iou_val, dice_score))

        if debug:
            if iou_val < 0.55:
                diff = gt_mask_image - mask
                plt.imshow(diff)
                plt.show()

        return iou_val, dice_score


import cog
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detic.modeling.utils import reset_cls_test
from detectron2.utils.visualizer import Visualizer


class DETIC(BaseImageSegmentation):
    def __init__(self, device='cpu', debug=False):
        super().__init__(device=device, debug=debug)
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(
            "/home/se86kimy/Dropbox/05_productive/01_code/15_ContrastiveFruitNeRF/CF-NeRF/cf_nerf/segmentation/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
        cfg.MODEL.WEIGHTS = '/home/se86kimy/Dropbox/05_productive/01_code/15_ContrastiveFruitNeRF/CF-NeRF/cf_nerf/segmentation/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
        # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
        # cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH  = 0.4
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = '/home/se86kimy/Dropbox/05_productive/01_code/15_ContrastiveFruitNeRF/CF-NeRF/cf_nerf/segmentation/Detic/datasets/metadata/lvis_v1_train_cat_info.json'
        self.predictor = DefaultPredictor(cfg)
        self.BUILDIN_CLASSIFIER = {
            'lvis': '/home/se86kimy/Dropbox/05_productive/01_code/15_ContrastiveFruitNeRF/CF-NeRF/cf_nerf/segmentation/Detic/datasets/metadata/lvis_v1_clip_a+cname.npy',
            'objects365': '/home/se86kimy/Dropbox/05_productive/01_code/15_ContrastiveFruitNeRF/CF-NeRF/cf_nerf/segmentation/Detic/datasets/metadata/o365_clip_a+cnamefix.npy',
            'openimages': '/home/se86kimy/Dropbox/05_productive/01_code/15_ContrastiveFruitNeRF/CF-NeRF/cf_nerf/segmentation/Detic/datasets/metadata/oid_clip_a+cname.npy',
            'coco': '/home/se86kimy/Dropbox/05_productive/01_code/15_ContrastiveFruitNeRF/CF-NeRF/cf_nerf/segmentation/Detic/datasets/metadata/coco_clip_a+cname.npy',
        }
        self.BUILDIN_METADATA_PATH = {
            'lvis': 'lvis_v1_val',
            'objects365': 'objects365_v2_val',
            'openimages': 'oid_val_expanded',
            'coco': 'coco_2017_val',
        }

        self.vocabulary = 'lvis'

        self.IOU = []
        self.DICE = []

    def run(self,
            image_path: Union[Path, str],
            text_prompt: Optional[str],
            output_filename,
            output_dir: Union[Path, str],
            box_threshold: float = 0.35,
            text_threshold: float = 0.35,
            nms_threshold: float = 0.5,
            flag_segmentation_image_debug=False,
            validation_path: Union[Path, str] = None):
        image = cv2.imread(image_path.__str__())
        metadata = MetadataCatalog.get(self.BUILDIN_METADATA_PATH[self.vocabulary])
        classifier = self.BUILDIN_CLASSIFIER[self.vocabulary]
        num_classes = len(metadata.thing_classes)
        reset_cls_test(self.predictor.model, classifier, num_classes)

        outputs = self.predictor(image)
        instance_id = 0

        mask = np.zeros_like(image[..., 0], dtype=np.uint16)
        mask_path = os.path.join(output_dir, output_filename.name)

        for current_instance_idx in range(outputs['instances'].__len__()):
            current_instance = outputs['instances'][current_instance_idx]

            if bool(current_instance.scores < 0.2):
                continue

            if isinstance(text_prompt, list):
                if metadata.thing_classes[int(current_instance.pred_classes)].lower() not in text_prompt:
                    print(metadata.thing_classes[int(current_instance.pred_classes)].lower())
                    continue
            elif isinstance(text_prompt, str):
                if metadata.thing_classes[int(current_instance.pred_classes)].lower() not in text_prompt:
                    print(metadata.thing_classes[int(current_instance.pred_classes)].lower())
                    continue

            instance_id += 1
            current_mask = outputs['instances'][current_instance_idx].pred_masks[0]
            mask[current_mask.cpu()] = instance_id

        if flag_segmentation_image_debug:
            v = Visualizer(image[:, :, ::-1], metadata)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            mask_path = os.path.join(output_dir, output_filename.name)
            annotated_image_path = os.path.join(output_dir, "overlay_" + output_filename.name)
            cv2.imwrite(str(annotated_image_path), out.get_image()[:, :, ::-1])

        # suffix_file = os.path.splitext(image_path)[-1]
        mask_path = os.path.splitext(mask_path)[0] + ".png"

        PIL.Image.fromarray(mask).save(mask_path)

        if validation_path:
            iou_val, dice_score = self.validate(validation_path, output_filename, pred_mask=mask)

            self.IOU.append(iou_val)
            self.DICE.append(dice_score)

        return {"image": mask, "path": mask_path}


class GroundedSAM(BaseImageSegmentation):
    def __init__(self, device='cpu', debug=False):
        super().__init__(device=device, debug=debug)

        import cf_nerf.segmentation as segmentation
        weights_base_path = Path(segmentation.__path__[0])

        self.model_config_path = weights_base_path / 'groundedSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        self.model_checkpoint_path = weights_base_path / 'groundedSAM/groundingdino_swint_ogc.pth'
        self.sam_checkpoint = weights_base_path / 'groundedSAM/sam_vit_h_4b8939.pth'
        self.sam_hq_checkpoint_h = weights_base_path / 'groundedSAM/sam_hq_vit_h.pth'
        self.sam_hq_checkpoint_l = weights_base_path / 'groundedSAM/sam_hq_vit_l.pth'

        self.SAM_ENCODER_VERSION = "vit_l"

        if self.SAM_ENCODER_VERSION == "vit_h":
            self.sam_hq_checkpoint = self.sam_hq_checkpoint_h
        elif self.SAM_ENCODER_VERSION == "vit_l":
            self.sam_hq_checkpoint = self.sam_hq_checkpoint_l
        else:
            raise ValueError("Wrong checkpoint for SAM encoder")

        self.IOU = []
        self.DICE = []

        self.model = self.load_model(self.model_config_path, self.model_checkpoint_path, self.device)

    def load_model(self, model_config_path, model_checkpoint_path, device):
        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=self.model_config_path,
                                          model_checkpoint_path=self.model_checkpoint_path)

        # Building SAM Model and SAM Predictor
        sam = sam_model_registry[self.SAM_ENCODER_VERSION](checkpoint=self.sam_hq_checkpoint)
        sam.to(device=DEVICE)
        self.sam_predictor = SamPredictor(sam)

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        return image

    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        ax.text(x0, y0, label)

    def run(self,
            image_path: Union[Path, str],
            text_prompt: Optional[str],
            output_filename,
            output_dir: Union[Path, str],
            box_threshold: float = 0.35,
            text_threshold: float = 0.35,
            nms_threshold: float = 0.5,
            flag_segmentation_image_debug=False,
            validation_path: Union[Path, str] = None):

        # Define threshold for segmentation
        BOX_THRESHOLD = box_threshold  # Default was 0.35
        TEXT_THRESHOLD = text_threshold  # Default was 0.35
        NMS_THRESHOLD = nms_threshold  # Default was 0.8

        image = self.load_image(image_path.__str__())

        if isinstance(text_prompt, str):
            CLASSES = [text_prompt]
        elif isinstance(text_prompt, list):
            CLASSES = text_prompt
        else:
            raise ValueError("Text prompt is wrong: {}".format(text_prompt))

        split_image_dino = True

        if split_image_dino:

            height, width, _ = image.shape
            half_height = height // 2
            half_width = width // 2

            overlap_x = int(0.2 * half_width)
            overlap_y = int(0.2 * half_height)

            part_1 = image[0:half_height + overlap_y, 0:half_width + overlap_x, :]
            part_2 = image[0:half_height + overlap_y, half_width - overlap_x:width, :]
            part_3 = image[half_height - overlap_y:height, 0:half_width + overlap_x, :]
            part_4 = image[half_height - overlap_y:height, half_width - overlap_x:width, :]

            image_list = [part_1, part_2, part_3, part_4]
            shift = [[0, 0],
                     [0, half_width - overlap_x],
                     [half_height - overlap_y, 0],
                     [half_height - overlap_y, half_width - overlap_x]]
            detection_list = []

            for sub_image, sub_shift in zip(image_list, shift):
                detections = self.grounding_dino_model.predict_with_classes(
                    image=sub_image,
                    classes=CLASSES,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD
                )

                detections.xyxy[:, [0, 2]] = detections.xyxy[:, [0, 2]] + sub_shift[1]
                detections.xyxy[:, [1, 3]] = detections.xyxy[:, [1, 3]] + sub_shift[0]

                detection_list.append(detections)

            # Fuse
            detections = detection_list[0]
            for detection in detection_list[1:]:
                detections.xyxy = np.vstack([detections.xyxy, detection.xyxy])
                detections.area_concat = np.concatenate([detections.area, detection.area])
                #detections.box_area = np.vstack([detections.box_area, detection.box_area])
                detections.class_id = np.concatenate([detections.class_id, detection.class_id]).astype(int)
                detections.confidence = np.concatenate([detections.confidence, detection.confidence])


        else:
            detections = self.grounding_dino_model.predict_with_classes(
                image=image,
                classes=CLASSES,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )
            detections.area_concat = detections.area

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()

        labels = [
            f"{CLASSES[int(class_id)]} {confidence:0.2f}"
            for confidence, class_id
            in zip(list(detections.confidence), list(detections.class_id))]
        if flag_segmentation_image_debug:
            # Display DINO bounding boxes
            annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)#, labels=labels)
            # #save the annotated grounding dino image
            cv2.imwrite(os.path.join(output_dir, "overlay_dino_" + output_filename.name), annotated_frame)

        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            boxes=torch.from_numpy(detections.xyxy),
            scores=torch.from_numpy(detections.confidence),
            iou_threshold=(float(NMS_THRESHOLD[0]) if isinstance(NMS_THRESHOLD, tuple) else NMS_THRESHOLD)
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        detections.area_concat = detections.area_concat[nms_idx]

        print(f"After NMS: {len(detections.xyxy)} boxes")

        # Prompting SAM with detected boxes
        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, logits = sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )
                index = np.argmax(scores)
                result_masks.append(masks[index])
            return np.array(result_masks)

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # Remove masks which are larger than 20% of the image
        for detection_idx in range(detections.__len__()):
            #if detections[detection_idx].area > image.shape[0] * image.shape[1] * 0.2:
            if detections.area_concat[detection_idx] > image.shape[0] * image.shape[1] * 0.2:
                # detections[detection_idx].mask = np.zeros_like(detections[detection_idx].mask, dtype=np.bool_)
                detections.mask[detection_idx] = np.zeros_like(detections[detection_idx].mask, dtype=np.bool_)
                detections.xyxy[detection_idx] = np.asarray([1, 0, 1, 0])

        # annotate image with detections
        mask_annotator = sv.MaskAnnotator()
        # annotated_image_rgb = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = mask_annotator.annotate(scene=np.zeros_like(image), detections=detections)

        if flag_segmentation_image_debug:
            box_annotator = sv.BoxAnnotator()
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)#, labels=labels)

        annotated_image_instance = np.zeros_like(image[..., 0], dtype=np.uint16)

        for idx, detection_idx in enumerate(np.flip(np.argsort(detections.area))):
            mask = detections.mask[detection_idx]
            # mask_instance = mask.astype(int) * (idx + 1)
            # plt.imshow(mask_instance)
            # plt.show()
            annotated_image_instance[mask] = (idx + 1)

        mask_path = os.path.join(output_dir, output_filename.name)
        mask = Image.fromarray(annotated_image_instance)

        # if '.JPG' in mask_path:
        #    mask_path = mask_path.replace('.JPG', '.PNG')

        # if '.PNG' in mask_path:
        #    mask_path = mask_path.replace('.JPG', '.PNG')

        # suffix_file = os.path.splitext(image_path)[-1]
        mask_path = os.path.splitext(mask_path)[0] + ".png"
        mask.save(mask_path)

        if flag_segmentation_image_debug:
            annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)#, labels=labels)
            annotated_image_path = os.path.join(output_dir, "overlay_" + output_filename.name)
            cv2.imwrite(annotated_image_path, annotated_image)

        if validation_path:
            iou_val, dice_score = self.validate(validation_path, output_filename, pred_mask=mask)

            self.IOU.append(iou_val)
            self.DICE.append(dice_score)

        return {"image": mask, "path": mask_path}


class SegmentImages(object):
    def __init__(self,
                 model: Literal['sam', 'detic'] = 'sam',
                 device: Literal['cpu', 'cuda'] = 'cpu',
                 debug: bool = False):
        self.model_type = model

        if self.model_type.lower() == 'sam':
            self.model = GroundedSAM(device=device, debug=debug)
        elif self.model_type.lower() == 'detic':
            self.model = DETIC(device=device, debug=debug)
        else:
            raise ValueError("Type {} is not implemented".format(self.model_type))

    def run(self,
            image_path,
            text_prompt,
            output_filename,
            output_dir,
            box_threshold: float,
            text_threshold: float,
            nms_threshold: float,
            flag_segmentation_image_debug: bool,
            validation_path: Union[Path, None] = None):
        return self.model.run(image_path,
                              text_prompt,
                              output_filename,
                              output_dir,
                              box_threshold,
                              text_threshold,
                              nms_threshold,
                              flag_segmentation_image_debug,
                              validation_path)


class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=5, aspect_ratio=0.3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled,
                                1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                  [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1],
                   vertex_transformed[4, :-1]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.show()


def convert_image_to_int16_array(image: Image.Image) -> np.ndarray:
    # Convert image to numpy array
    image_array = np.array(image)

    # Reshape image array to 2D array of RGB colors
    reshaped_array = image_array.reshape(-1, 1)

    # Get unique colors and map them to unique integer IDs starting from 0
    unique_colors, ids = np.unique(reshaped_array, axis=0, return_inverse=True)

    # Convert the image array to the corresponding ID array
    int16_array = ids.reshape(image_array.shape[:2]).astype(np.int16)

    return int16_array


def copy_images(
        data: Path,
        image_dir: Path,
        image_prefix: str = "frame_",
        verbose: bool = False,
        keep_image_dir: bool = False,
        crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        num_downscales: int = 0,
        same_dimensions: bool = True,
) -> OrderedDict[Path, Path]:
    """Copy images from a directory to a new directory.

    Args:
        data: Path to the directory of images.
        image_dir: Path to the output directory.
        image_prefix: Prefix for the image filenames.
        verbose: If True, print extra logging.
        crop_factor: Portion of the image to crop. Should be in [0,1] (top, bottom, left, right)
        keep_image_dir: If True, don't delete the output directory if it already exists.
    Returns:
        The mapping from the original filenames to the new ones.
    """
    with status(msg="[bold yellow]Copying images...", spinner="bouncingBall", verbose=verbose):
        image_paths = process_data_utils.list_images(data)

        if len(image_paths) == 0:
            CONSOLE.log("[bold red]:skull: No usable images in the data folder.")
            sys.exit(1)

        copied_images = process_data_utils.copy_images_list(
            image_paths=image_paths,
            image_dir=image_dir,
            crop_factor=crop_factor,
            verbose=verbose,
            image_prefix=image_prefix,
            keep_image_dir=keep_image_dir,
            num_downscales=num_downscales,
            same_dimensions=same_dimensions,
            nearest_neighbor=True
        )
        return OrderedDict((original_path, new_path) for original_path, new_path in zip(image_paths, copied_images))


@dataclass
class CFNeRFDatasetToNerfstudioDataset(ColmapConverterToNerfstudioDataset):
    num_downscales: int = 1
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3 will downscale the
       images by 2x, 4x, and 8x."""
    gpu: bool = True
    """If True, use GPU."""
    metadata: str = 'transforms.json'
    crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """Portion of the image to crop. All values should be in [0,1]. (top, bottom, left, right)"""
    same_dimensions: bool = True
    """Whether to assume all images are same dimensions and so to use fast downscaling with no autorotation."""
    compute_instance_mask: bool = True
    """Compute instance mask."""
    instance_model: Literal['SAM', 'DETIC', 'sam', 'detic'] = 'sam'
    """Which model to use. SAM or DETIC."""
    segmentation_class: Union[str, list, None] = field(
        default_factory=lambda: ['fruit', 'apple', 'pomegranate', 'peach'])
    text_threshold: float = 0.25
    """Text threshold for DINO/SAM"""
    box_threshold: float = 0.3
    """Box threshold for DINO/SAM"""
    nms_threshold: float = 0.3
    """NMS for fusing boxes"""
    semantics_gt: Union[None, str] = None
    save_debug_images: bool = True
    use_colmap: bool = True
    """If True, use COLMAP"""

    @property
    def rgb_image_path(self) -> Path:
        return self.data / 'images'

    @property
    def semantics_path(self) -> Path:
        if self.compute_instance_mask:
            path = self.output_instance_path
        else:
            path = self.data / 'semantics'
        return path

    # @property
    # def semantics_gt_path(self) -> Path:
    #    path = self.data / 'semantics_gt'
    #    return path

    @property
    def metadata_path(self) -> Path:
        if self.use_colmap:
            p =self.output_dir / self.metadata
        else:
            p = self.data / self.metadata
        return p

    @property
    def output_path(self) -> Path:
        return self.output_dir

    @property
    def output_image_path(self) -> Path:
        return self.output_path / 'images'

    @property
    def output_instance_path(self) -> Path:
        # if self.compute_instance_mask:
        #    if self.instance_model.lower() == 'sam':
        #        path = self.output_path / 'semantics_sam'
        #    elif self.instance_model.lower() == 'detic':
        #        path = self.output_path / 'semantics_detic'
        #    else:
        #        raise ValueError(f'Unknown instance model: {self.instance_model}')
        # else:
        path = self.output_path / 'semantics'
        return path

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""
        summary_log = []
        image_rename_map: Optional[dict[str, str]] = None

        if self.semantics_gt:
            self.semantics_gt = Path(self.semantics_gt)

        if self.compute_instance_mask:
            segmentor = SegmentImages(model=self.instance_model, device=DEVICE)

        # image_filenames, num_orig_images = process_data_utils.get_image_filenames(self.output_image_path)

        if self.colmap_model_path != ColmapConverterToNerfstudioDataset.default_colmap_path():
            if self.use_colmap:
                raise RuntimeError("The --colmap-model-path can only be used when --skip-colmap is not set.")
            if not (self.output_dir / self.colmap_model_path).exists():
                raise RuntimeError(f"The colmap-model-path {self.output_dir / self.colmap_model_path} does not exist.")

        image_rename_map_paths = process_data_utils.copy_images(
            self.rgb_image_path,
            image_dir=self.output_image_path,
            crop_factor=self.crop_factor,
            image_prefix="frame_train_" if self.eval_data is not None else "frame_",
            verbose=self.verbose,
            num_downscales=self.num_downscales,
            same_dimensions=self.same_dimensions,
            keep_image_dir=False,
        )

        image_rename_map = dict(
            #(a.relative_to(self.data).as_posix(), b.name) for a, b in image_rename_map_paths.items()
            (a.name, b.name) for a, b in image_rename_map_paths.items()
        )
        num_frames_rgb = len(image_rename_map)
        summary_log.append(f"Starting with {num_frames_rgb} images")

        # Run COLMAP
        if self.use_colmap:
            self._run_colmap()
            # Colmap uses renamed images
            # COLMAP to  transform.json
            summary_log += self._save_transforms(
                num_frames_rgb,
                None,
                None,
                image_rename_map=None,
            )


        self.output_instance_path.mkdir(exist_ok=True)

        # Compute SAM or Detic here!
        if self.compute_instance_mask:
            for image_path in image_rename_map_paths:
                mask = segmentor.run(image_path=image_path,
                                     text_prompt=self.segmentation_class,
                                     output_filename=image_rename_map_paths[image_path],
                                     text_threshold=self.text_threshold,
                                     box_threshold=self.box_threshold,
                                     nms_threshold=self.nms_threshold,
                                     output_dir=self.output_instance_path,
                                     flag_segmentation_image_debug=self.save_debug_images,
                                     validation_path=self.semantics_gt)
                image_rename_map_paths[Path(image_path)] = Path(mask['path'])

            if self.semantics_gt:
                print("IOU: ", np.mean(segmentor.model.IOU))
                print("DICE: ", np.mean(segmentor.model.DICE))
            del (segmentor)
        # Copy semantics from original destination or computed. Down sample images
        image_rename_map_paths_instance = copy_images(
            self.semantics_path,
            image_dir=self.output_instance_path,
            crop_factor=self.crop_factor,
            image_prefix="frame_train_" if self.eval_data is not None else "frame_",
            verbose=self.verbose,
            num_downscales=self.num_downscales,
            same_dimensions=self.same_dimensions,
            keep_image_dir=False,
        )

        # image_rename_map_instance = dict(
        #    (a.relative_to(self.output_dir).as_posix(), b.name) for a, b in image_rename_map_paths_instance.items()
        # )
        # num_frames_instance = len(image_rename_map_instance)
        # summary_log.append(f"Starting with {num_frames_instance} semantic images")

        with open(self.metadata_path) as f:
            self.metadata_file = json.load(f)

        # img_h, img_w = np.array(Image.open(next(iter(image_rename_map_paths)))).shape[:2]

        if False:

            if True:
                visualizer = CameraPoseVisualizer([-10, 10], [-10, 10], [0, 10])

                for p in P:
                    visualizer.extrinsic2pyramid(p, 'c', 1)
                visualizer.show()

            self.transform = {
                "camera_angle_x": 2 * np.arctan(img_w / K[0, 0]),
                "camera_angle_y": 2 * np.arctan(img_h / K[1, 1]),
                "fl_x": K[0, 0],
                "fl_y": K[1, 1],
                "k1": 0.0,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "cx": K[0, 2],
                "cy": K[1, 2],
                "w": img_w,
                "h": img_h,
                "aabb_scale": 16,
                'frames': [],
                "semantics": [
                    "stuff",
                    "test"
                ]
            }
            for idx, image_path in enumerate(image_rename_map.keys()):
                element = {
                    "file_path": "images/{}".format(image_rename_map[image_path]),
                    "transform_matrix": [
                        P[idx][0].tolist(), P[idx][1].tolist(), P[idx][2].tolist(), P[idx][3].tolist()
                    ],
                    "semantic_path": "semantics/{}".format(image_rename_map[image_path])
                }

                self.transform['frames'].append(element)

        if "colmap_im_id" in self.metadata_file['frames'][0].keys():
            for idx, frame in enumerate(self.metadata_file['frames']):
                file_name = frame['file_path'].split("/")[-1]
                semanitic_file_name = os.path.splitext(frame['file_path'])[0] + ".png"
                frame['semantic_path'] = "{}/{}".format(self.semantics_path.parts[-1], semanitic_file_name)
        else:
            for idx, image_path in enumerate(image_rename_map.keys()):
                file_name = image_rename_map[image_path]
                semanitic_file_name = os.path.splitext(file_name)[0] + ".png"

                self.metadata_file['frames'][idx]['semantic_path'] = "{}/{}".format(self.semantics_path.parts[-1],
                                                                                    file_name)
                self.metadata_file['frames'][idx]['file_path'] = "{}/{}".format('images', semanitic_file_name)

        json.dump(self.metadata_file, open(self.output_path / 'transforms.json', 'w+'), indent=4)

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.log(summary)


"""
for idx, frame in enumerate(self.metadata_file['frames']):
    file_name = self.metadata_file_mapping[frame['file_path'].split("/")[-1]]
    semanitic_file_name = os.path.splitext(file_name)[0] + ".png"
    frame['semantic_path'] = "{}/{}".format(self.semantics_path.parts[-1], semanitic_file_name)
    
with open("/home/se86kimy/Dropbox/07_data/CF-NeRF/FUJI/file_mapping.json") as f:
    self.metadata_file_mapping = json.load(f)
"""
