import os
import os.path as osp
import torch
import numpy as np
import torch.cuda.nvtx as nvtx

from mmengine.logging import MMLogger
from mmengine.dist import is_main_process

from mmdet3d.evaluation import InstanceSegMetric
from mmdet3d.evaluation.metrics import SegMetric
from mmdet3d.registry import METRICS
from mmdet3d.evaluation import panoptic_seg_eval, seg_eval
from .instance_seg_eval import instance_seg_eval


@METRICS.register_module()
class UnifiedSegMetric(SegMetric):
    """Metric for instance, semantic, and panoptic evaluation.
    The order of classes must be [stuff classes, thing classes, unlabeled].

    Args:
        thing_class_inds (List[int]): Ids of thing classes.
        stuff_class_inds (List[int]): Ids of stuff classes.
        min_num_points (int): Minimal size of mask for panoptic segmentation.
        id_offset (int): Offset for instance classes.
        sem_mapping (List[int]): Semantic class to gt id.
        inst_mapping (List[int]): Instance class to gt id.
        metric_meta (Dict): Analogue of dataset meta of SegMetric. Keys:
            `label2cat` (Dict[int, str]): class names,
            `ignore_index` (List[int]): ids of semantic categories to ignore,
            `classes` (List[str]): class names.
        logger_keys (List[Tuple]): Keys for logger to save; of len 3:
            semantic, instance, and panoptic.
    """

    def __init__(self,
                 thing_class_inds,
                 stuff_class_inds,
                 min_num_points,
                 id_offset,
                 sem_mapping,   
                 inst_mapping,
                 metric_meta,
                 dump_scene_preds=False,
                 dump_dir=None,
                 logger_keys=[('miou',),
                              ('all_ap', 'all_ap_50%', 'all_ap_25%'), 
                              ('pq',)],
                 **kwargs):
        self.thing_class_inds = thing_class_inds
        self.stuff_class_inds = stuff_class_inds
        self.min_num_points = min_num_points
        self.id_offset = id_offset
        self.metric_meta = metric_meta
        self.logger_keys = logger_keys
        self.sem_mapping = np.array(sem_mapping)
        self.inst_mapping = np.array(inst_mapping)
        self.dump_scene_preds = dump_scene_preds
        self.dump_dir = dump_dir
        super().__init__(**kwargs)

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        self.valid_class_ids = self.dataset_meta['seg_valid_class_ids']
        label2cat = self.metric_meta['label2cat']
        ignore_index = self.metric_meta['ignore_index']
        classes = self.metric_meta['classes']
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]
        num_stuff_cls = len(stuff_classes)

        gt_semantic_masks_inst_task = []
        gt_instance_masks_inst_task = []
        pred_instance_masks_inst_task = []
        pred_instance_labels = []
        pred_instance_scores = []

        gt_semantic_masks_sem_task = []
        pred_semantic_masks_sem_task = []

        gt_masks_pan = []
        pred_masks_pan = []

        dump_scene_preds = self.dump_scene_preds and is_main_process()
        if dump_scene_preds and self.dump_dir is not None:
            os.makedirs(self.dump_dir, exist_ok=True)

        for idx, (eval_ann, single_pred_results) in enumerate(results):
            
            if self.metric_meta['dataset_name'] == 'S3DIS':
                pan_gt = {}
                pan_gt['pts_semantic_mask'] = eval_ann['pts_semantic_mask']
                pan_gt['pts_instance_mask'] = \
                                    eval_ann['pts_instance_mask'].copy()

                for stuff_cls in self.stuff_class_inds:
                    pan_gt['pts_instance_mask'][\
                        pan_gt['pts_semantic_mask'] == stuff_cls] = \
                                    np.max(pan_gt['pts_instance_mask']) + 1

                pan_gt['pts_instance_mask'] = np.unique(
                                                pan_gt['pts_instance_mask'],
                                                return_inverse=True)[1]
                gt_masks_pan.append(pan_gt)
            else:
                gt_masks_pan.append(eval_ann)
            
            pred_masks_pan.append({
                'pts_instance_mask': \
                    single_pred_results['pts_instance_mask'][1],
                'pts_semantic_mask': \
                    single_pred_results['pts_semantic_mask'][1]
            })

            gt_semantic_masks_sem_task.append(eval_ann['pts_semantic_mask'])            
            pred_semantic_masks_sem_task.append(
                single_pred_results['pts_semantic_mask'][0])

            if self.metric_meta['dataset_name'] == 'S3DIS':
                gt_semantic_masks_inst_task.append(eval_ann['pts_semantic_mask'])
                gt_instance_masks_inst_task.append(eval_ann['pts_instance_mask'])  
            else:
                sem_mask, inst_mask = self.map_inst_markup(
                    eval_ann['pts_semantic_mask'].copy(), 
                    eval_ann['pts_instance_mask'].copy(), 
                    self.valid_class_ids[num_stuff_cls:],
                    num_stuff_cls)
                gt_semantic_masks_inst_task.append(sem_mask)
                gt_instance_masks_inst_task.append(inst_mask)           
            
            pred_instance_masks_inst_task.append(
                torch.tensor(single_pred_results['pts_instance_mask'][0]))
            pred_instance_labels.append(
                torch.tensor(single_pred_results['instance_labels']))
            pred_instance_scores.append(
                torch.tensor(single_pred_results['instance_scores']))

            if dump_scene_preds and self.dump_dir is not None:
                with nvtx.range("dump_scene_preds"):
                    def _maybe_basename(val):
                        if val is None:
                            return None
                        if isinstance(val, (list, tuple)) and len(val) == 1:
                            val = val[0]
                        val = str(val)
                        # strip extension if it looks like a path/filename
                        if '/' in val or '\\' in val or '.' in val:
                            val = osp.splitext(osp.basename(val))[0]
                        return val

                    candidates = []
                    if isinstance(eval_ann, dict):
                        metainfo = eval_ann.get('metainfo', {}) if isinstance(eval_ann.get('metainfo', {}), dict) else {}
                        for key in ['scene_id', 'scan_id', 'lidar_path', 'pts_path', 'filename', 'sample_idx']:
                            cand = _maybe_basename(eval_ann.get(key))
                            if cand:
                                candidates.append(cand)
                        meta_cand = _maybe_basename(metainfo.get('scene_id') or metainfo.get('sample_idx'))
                        if meta_cand:
                            candidates.append(meta_cand)
                        lidar_info = eval_ann.get('lidar_points') or {}
                        if isinstance(lidar_info, dict):
                            lp_cand = _maybe_basename(lidar_info.get('lidar_path') or lidar_info.get('pts_path'))
                            if lp_cand:
                                candidates.append(lp_cand)
                        sem_path = _maybe_basename(eval_ann.get('pts_semantic_mask_path'))
                        if sem_path:
                            candidates.append(sem_path)

                    scene_id = next((c for c in candidates if not str(c).isdigit()), None)
                    if scene_id is None and candidates:
                        scene_id = candidates[0]
                    if scene_id is None:
                        scene_id = str(idx)
                    scene_id = str(scene_id)
                    # Build id_offset-encoded panoptic ids:
                    #  - stuff points keep their semantic class id
                    #  - thing points store (id_offset + instance_id)
                    pan_sem = np.asarray(single_pred_results['pts_semantic_mask'][1])
                    pan_inst = np.asarray(single_pred_results['pts_instance_mask'][1])
                    stuff_mask = np.isin(pan_sem, self.stuff_class_inds)
                    panoptic_pred = pan_sem.copy()
                    panoptic_pred[~stuff_mask] = self.id_offset + pan_inst[~stuff_mask]

                    # Persist everything needed for post-hoc instance AP:
                    #  - instance_masks: boolean [n_inst, n_points]
                    #  - instance_labels / instance_scores
                    #  - panoptic_pred / panoptic_semantic for visualization
                    instance_masks = np.asarray(
                        single_pred_results['pts_instance_mask'][0]
                    ).astype(np.uint8)  # uint8 saves space vs bool

                    save_obj = {
                        'scene_id': scene_id,
                        'panoptic_pred': panoptic_pred.astype(np.int64),
                        'panoptic_semantic': pan_sem.astype(np.int64),
                        'instance_masks': instance_masks,
                        'instance_labels': single_pred_results['instance_labels'],
                        'instance_scores': single_pred_results['instance_scores'],
                    }
                    save_path = osp.join(self.dump_dir, f'{scene_id}.pth')
                    torch.save(save_obj, save_path)
                    print(f"[dump_scene_preds] saved {save_path}")

        ret_pan = panoptic_seg_eval(
            gt_masks_pan, pred_masks_pan, classes, thing_classes,
            stuff_classes, self.min_num_points, self.id_offset,
            label2cat, ignore_index, logger)

        ret_sem = seg_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            label2cat,
            ignore_index[0],
            logger=logger)

        if self.metric_meta['dataset_name'] == 'S3DIS':
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids,
                class_labels=classes[:-1],
                logger=logger)
        else:
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids[num_stuff_cls:],
                class_labels=classes[num_stuff_cls:-1],
                logger=logger)

        metrics = dict()
        for ret, keys in zip((ret_sem, ret_inst, ret_pan), self.logger_keys):
            for key in keys:
                metrics[key] = ret[key]
        return metrics

    def map_inst_markup(self,
                        pts_semantic_mask,
                        pts_instance_mask,
                        valid_class_ids,
                        num_stuff_cls):
        """Map gt instance and semantic classes back from panoptic annotations.

        Args:
            pts_semantic_mask (np.array): of shape (n_raw_points,)
            pts_instance_mask (np.array): of shape (n_raw_points.)
            valid_class_ids (Tuple): of len n_instance_classes
            num_stuff_cls (int): number of stuff classes
        
        Returns:
            Tuple:
                np.array: pts_semantic_mask of shape (n_raw_points,)
                np.array: pts_instance_mask of shape (n_raw_points,)
        """
        pts_instance_mask -= num_stuff_cls
        pts_instance_mask[pts_instance_mask < 0] = -1
        pts_semantic_mask -= num_stuff_cls
        pts_semantic_mask[pts_instance_mask == -1] = -1

        mapping = np.array(list(valid_class_ids) + [-1])
        pts_semantic_mask = mapping[pts_semantic_mask]
        
        return pts_semantic_mask, pts_instance_mask


@METRICS.register_module()
class InstanceSegMetric_(InstanceSegMetric):
    """The only difference with InstanceSegMetric is that following ScanNet
    evaluator we accept instance prediction as a boolean tensor of shape
    (n_points, n_instances) instead of integer tensor of shape (n_points, ).

    For this purpose we only replace instance_seg_eval call.
    """

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        self.classes = self.dataset_meta['classes']
        self.valid_class_ids = self.dataset_meta['seg_valid_class_ids']

        gt_semantic_masks = []
        gt_instance_masks = []
        pred_instance_masks = []
        pred_instance_labels = []
        pred_instance_scores = []

        for eval_ann, single_pred_results in results:
            gt_semantic_masks.append(eval_ann['pts_semantic_mask'])
            gt_instance_masks.append(eval_ann['pts_instance_mask'])
            pred_instance_masks.append(
                single_pred_results['pts_instance_mask'])
            pred_instance_labels.append(single_pred_results['instance_labels'])
            pred_instance_scores.append(single_pred_results['instance_scores'])

        ret_dict = instance_seg_eval(
            gt_semantic_masks,
            gt_instance_masks,
            pred_instance_masks,
            pred_instance_labels,
            pred_instance_scores,
            valid_class_ids=self.valid_class_ids,
            class_labels=self.classes,
            logger=logger)

        return ret_dict
