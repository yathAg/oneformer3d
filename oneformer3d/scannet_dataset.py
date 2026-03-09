from os import path as osp
import numpy as np
import random

from mmdet3d.datasets.scannet_dataset import ScanNetSegDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class ScanNetSegDataset_(ScanNetSegDataset):
    """We just add super_pts_path."""

    def get_scene_idxs(self, *args, **kwargs):
        """Compute scene_idxs for data sampling."""
        return np.arange(len(self)).astype(np.int32)

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info['super_pts_path'] = osp.join(
            self.data_prefix.get('sp_pts_mask', ''), info['super_pts_path'])

        info = super().parse_data_info(info)

        # Ensure eval_ann_info carries identifiers for dumping
        eval_ann = info.get('eval_ann_info', {}) if isinstance(
            info.get('eval_ann_info', {}), dict) else {}
        eval_ann.setdefault('scene_id', info.get('scene_id'))
        eval_ann.setdefault('scan_id', info.get('scan_id'))
        eval_ann.setdefault('sample_idx', info.get('sample_idx'))
        eval_ann.setdefault('lidar_path', info.get('lidar_path'))
        eval_ann.setdefault('pts_path', info.get('pts_path'))
        eval_ann.setdefault('filename', info.get('filename'))
        eval_ann.setdefault('metainfo', info.get('metainfo'))
        eval_ann.setdefault('lidar_points', info.get('lidar_points'))
        eval_ann.setdefault('pts_semantic_mask_path',
                            info.get('pts_semantic_mask_path'))
        info['eval_ann_info'] = eval_ann

        return info


@DATASETS.register_module()
class ScanNet200SegDataset_(ScanNetSegDataset_):
    # IMPORTANT: the floor and chair categories are swapped.
    METAINFO = {
    'classes': ('wall', 'floor', 'chair', 'table', 'door', 'couch', 'cabinet',
                'shelf', 'desk', 'office chair', 'bed', 'pillow', 'sink',
                'picture', 'window', 'toilet', 'bookshelf', 'monitor',
                'curtain', 'book', 'armchair', 'coffee table', 'box',
                'refrigerator', 'lamp', 'kitchen cabinet', 'towel', 'clothes',
                'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion',
                'plant', 'ceiling', 'bathtub', 'end table', 'dining table',
                'keyboard', 'bag', 'backpack', 'toilet paper', 'printer',
                'tv stand', 'whiteboard', 'blanket', 'shower curtain',
                'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe',
                'computer tower', 'bottle', 'bin', 'ottoman', 'bench', 'board',
                'washing machine', 'mirror', 'copier', 'basket', 'sofa chair',
                'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person',
                'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate',
                'blackboard', 'piano', 'suitcase', 'rail', 'radiator',
                'recycling bin', 'container', 'wardrobe', 'soap dispenser',
                'telephone', 'bucket', 'clock', 'stand', 'light',
                'laundry basket', 'pipe', 'clothes dryer', 'guitar',
                'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle',
                'ladder', 'bathroom stall', 'shower wall', 'cup', 'jacket',
                'storage bin', 'coffee maker', 'dishwasher',
                'paper towel roll', 'machine', 'mat', 'windowsill', 'bar',
                'toaster', 'bulletin board', 'ironing board', 'fireplace',
                'soap dish', 'kitchen counter', 'doorframe',
                'toilet paper dispenser', 'mini fridge', 'fire extinguisher',
                'ball', 'hat', 'shower curtain rod', 'water cooler',
                'paper cutter', 'tray', 'shower door', 'pillar', 'ledge',
                'toaster oven', 'mouse', 'toilet seat cover dispenser',
                'furniture', 'cart', 'storage container', 'scale',
                'tissue box', 'light switch', 'crate', 'power outlet',
                'decoration', 'sign', 'projector', 'closet door',
                'vacuum cleaner', 'candle', 'plunger', 'stuffed animal',
                'headphones', 'dish rack', 'broom', 'guitar case',
                'range hood', 'dustpan', 'hair dryer', 'water bottle',
                'handicap bar', 'purse', 'vent', 'shower floor',
                'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock',
                'music stand', 'projector screen', 'divider',
                'laundry detergent', 'bathroom counter', 'object',
                'bathroom vanity', 'closet wall', 'laundry hamper',
                'bathroom stall door', 'ceiling light', 'trash bin',
                'dumbbell', 'stair rail', 'tube', 'bathroom cabinet',
                'cd case', 'closet rod', 'coffee kettle', 'structure',
                'shower head', 'keyboard piano', 'case of water bottles',
                'coat rack', 'storage organizer', 'folded chair', 'fire alarm',
                'power strip', 'calendar', 'poster', 'potted plant', 'luggage',
                'mattress'),
    # the valid ids of segmentation annotations
    'seg_valid_class_ids': (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22,
        23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44,
        45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65,
        66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86,
        87, 88, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
        106, 107, 110, 112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131,
        132, 134, 136, 138, 139, 140, 141, 145, 148, 154,155, 156, 157, 159,
        161, 163, 165, 166, 168, 169, 170, 177, 180, 185, 188, 191, 193, 195,
        202, 208, 213, 214, 221, 229, 230, 232, 233, 242, 250, 261, 264, 276,
        283, 286, 300, 304, 312, 323, 325, 331, 342, 356, 370, 392, 395, 399,
        408, 417, 488, 540, 562, 570, 572, 581, 609, 748, 776, 1156, 1163,
        1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175,
        1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188,
        1189, 1190, 1191),
    'seg_all_class_ids': tuple(range(1, 1358)),
    'palette': [random.sample(range(0, 255), 3) for i in range(200)]}


@DATASETS.register_module()
class ScanNetPPSegDataset_(ScanNetSegDataset_):
    METAINFO = {
        'classes': (
            'wall', 'ceiling', 'floor', 'table', 'door', 'ceiling lamp',
            'cabinet', 'blinds', 'curtain', 'chair', 'storage cabinet',
            'office chair', 'bookshelf', 'whiteboard', 'window', 'box',
            'window frame', 'monitor', 'shelf', 'doorframe', 'pipe', 'heater',
            'kitchen cabinet', 'sofa', 'windowsill', 'bed', 'shower wall',
            'trash can', 'book', 'plant', 'blanket', 'tv', 'computer tower',
            'kitchen counter', 'refrigerator', 'jacket', 'electrical duct',
            'sink', 'bag', 'picture', 'pillow', 'towel', 'suitcase',
            'backpack', 'crate', 'keyboard', 'rack', 'toilet', 'paper',
            'printer', 'poster', 'painting', 'microwave', 'board', 'shoes',
            'socket', 'bottle', 'bucket', 'cushion', 'basket', 'shoe rack',
            'telephone', 'file folder', 'cloth', 'blind rail', 'laptop',
            'plant pot', 'exhaust fan', 'cup', 'coat hanger', 'light switch',
            'speaker', 'table lamp', 'air vent', 'clothes hanger', 'kettle',
            'smoke detector', 'container', 'power strip', 'slippers',
            'paper bag', 'mouse', 'cutting board', 'toilet paper',
            'paper towel', 'pot', 'clock', 'pan', 'tap', 'jar',
            'soap dispenser', 'binder', 'bowl', 'tissue box',
            'whiteboard eraser', 'toilet brush', 'spray bottle', 'headphones',
            'stapler', 'marker'),
        'seg_valid_class_ids': tuple(range(1, 101)),
        'seg_all_class_ids': tuple(range(1, 101)),
        'palette': [random.sample(range(0, 255), 3) for i in range(100)]
    }


@DATASETS.register_module()
class ScanNetPPSegDatasetExt_(ScanNetSegDataset_):
    METAINFO = {
        'classes': (
            'wall', 'ceiling', 'floor', 'shower wall', 'bathroom wall',
            'tiled wall', 'object', 'box', 'chair', 'ceiling lamp', 'book',
            'socket', 'table', 'window', 'door', 'bottle', 'light switch',
            'pipe', 'cabinet', 'monitor', 'shelf', 'paper', 'heater',
            'office chair', 'pillow', 'doorframe', 'window frame', 'shoes',
            'trash can', 'picture', 'cup', 'plant', 'keyboard', 'bag',
            'windowsill', 'clothes', 'curtain', 'mouse', 'jacket',
            'door frame', 'towel', 'smoke detector', 'objects',
            'kitchen cabinet', 'power sockets', 'backpack', 'plant pot',
            'blanket', 'whiteboard', 'blinds', 'poster', 'sink',
            'computer tower', 'electrical duct', 'decoration', 'bed',
            'cable raceway', 'structure', 'wall lamp', 'laptop',
            'storage cabinet', 'power strip', 'suitcase', 'bookshelf',
            'telephone', 'jar', 'cloth', 'sofa', 'basket', 'wardrobe',
            'speaker', 'crate', 'toilet paper', 'stool', 'cable duct',
            'faucet', 'table lamp', 'switch', 'bowl', 'kitchen counter',
            'cable', 'extension cord', 'slippers', 'bucket',
            'electrical control panel', 'binder', 'light switches',
            'power outlet', 'painting', 'mattress', 'rug', 'plate',
            'toilet', 'notebook', 'microwave', 'shelf rail', 'board', 'tv',
            'whiteboard eraser', 'refrigerator', 'coax outlet', 'file folder',
            'router', 'cushion', 'cutting board', 'bedside table', 'lamp',
            'phone charger', 'kettle', 'paper bag', 'paper towel', 'carpet',
            'headphones', 'soap dispenser', 'machine', 'pot', 'vase',
            'trolley', 'tap', 'umbrella', 'tray', 'curtain rod',
            'kitchen towel', 'sponge', 'oven', 'mirror', 'container',
            'folder', 'pedestal fan', 'lab equipment', 'spray bottle',
            'storage rack', 'tube', 'curtain rail',
            'paper towel dispenser', 'laundry basket', 'luggage', 'pillar',
            'bottle crate', 'counter', 'intercom', 'standing lamp',
            'photograph', 'toilet brush', 'exhaust fan', 'heat pipe',
            'note', 'printer', 'electric kettle', 'vent', 'floor lamp',
            'coffee maker', 'opaque window panel', 'coat', 'file binder',
            'frying pan', 'shirt', 'pc', 'light', 'mop', 'roller blinds',
            'air vent', 'ceiling panel', 'tissue box', 'desk lamp',
            'shampoo bottle', 'bottles', 'clock', 'power outlets', 'broom',
            'pen holder', 'projector', 'stove', 'ball', 'stove top',
            'thermos', 'whiteboard marker', 'yoga mat', 'cooking pot',
            'vacuum cleaner', 'fire extinguisher', 'plastic bag',
            'toilet flush button', 'magazine', 'panel', 'toilet paper roll',
            'valve', 'blackboard', 'desk divider', 'candle', 'coffee table',
            'dumbbell', 'laptop charger', 'coat hanger', 'picture frame',
            'dustbin', 'knife', 'pan', 'power adapter',
            'reusable shopping bag'),
        'seg_valid_class_ids': tuple(range(1, 201)),
        'seg_all_class_ids': tuple(range(1, 201)),
        'palette': [random.sample(range(0, 255), 3) for i in range(200)]
    }
