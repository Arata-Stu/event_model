"""
Functions to display events and boxes
Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import print_function

from tqdm import tqdm  # これが必要
import bbox_visualizer as bbv
import cv2
import numpy as np
import torch
import lightning as pl
from einops import rearrange, reduce

from utils.padding import InputPadderFromShape
from data.utils.types import DatasetMode, DataType
from data.utils.types import DataType
from data.genx_utils.labels import ObjectLabels
from modules.utils.detection import RNNStates
from models.layers.yolox.utils.boxes import postprocess

LABELMAP_GEN1 = ("car", "pedestrian")
LABELMAP_GEN4 = ('pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light')
LABELMAP_GEN4_SHORT = ('pedestrian', 'two wheeler', 'car')

## 0~7まで定義しておく
classid2colors = {
    0: (0, 0, 255),  # ped -> blue (rgb)
    1: (0, 255, 255),  # 2-wheeler cyan (rgb)
    2: (255, 255, 0),  # car -> yellow (rgb)
    3: (255, 0, 0),  # truck -> red (rgb)
    4: (255, 0, 255),  # bus -> magenta (rgb)
    5: (0, 255, 0),  # traffic sign -> green (rgb)
    6: (0, 0, 0),  # traffic light -> black (rgb)
    7: (255, 255, 255),  # other -> white (rgb)
}

dataset2labelmap = {
    "gen1": LABELMAP_GEN1,
    "gen4": LABELMAP_GEN4
}

dataset2scale = {
    "gen1": 1,
    "gen4": 1
}

dataset2size = {
    "gen1": (304*1, 240*1),
    "gen4": (640*1, 360*1),
}

def ev_repr_to_img(x: np.ndarray):
    ch, ht, wd = x.shape[-3:]
    assert ch > 1 and ch % 2 == 0
    ev_repr_reshaped = rearrange(x, '(posneg C) H W -> posneg C H W', posneg=2)
    img_neg = np.asarray(reduce(ev_repr_reshaped[0], 'C H W -> H W', 'sum'), dtype='int32')
    img_pos = np.asarray(reduce(ev_repr_reshaped[1], 'C H W -> H W', 'sum'), dtype='int32')
    img_diff = img_pos - img_neg
    img = 127 * np.ones((ht, wd, 3), dtype=np.uint8)
    img[img_diff > 0] = 255
    img[img_diff < 0] = 0
    return img


def make_binary_histo(events, img=None, width=304, height=240):
    """
    simple display function that shows negative events as blacks dots and positive as white one
    on a gray background
    args :
        - events structured numpy array
        - img (numpy array, height x width x 3) optional array to paint event on.
        - width int
        - height int
    return:
        - img numpy array, height x width x 3)
    """
    if img is None:
        img = 127 * np.ones((height, width, 3), dtype=np.uint8)
    else:
        # if an array was already allocated just paint it grey
        img[...] = 127
    if events.size:
        assert events['x'].max() < width, "out of bound events: x = {}, w = {}".format(events['x'].max(), width)
        assert events['y'].max() < height, "out of bound events: y = {}, h = {}".format(events['y'].max(), height)

        img[events['y'], events['x'], :] = 255 * events['p'][:, None]
    return img


def draw_bboxes_bbv(img, boxes, dataset_name: str) -> np.ndarray:
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    labelmap=dataset2labelmap[dataset_name]
    scale_multiplier = dataset2scale[dataset_name]

    add_score = True
    ht, wd, ch = img.shape
    dim_new_wh = (int(wd * scale_multiplier), int(ht * scale_multiplier))
    if scale_multiplier != 1:
        img = cv2.resize(img, dim_new_wh, interpolation=cv2.INTER_AREA)
    for i in range(boxes.shape[0]):
        pt1 = (int(boxes['x'][i]), int(boxes['y'][i]))
        size = (int(boxes['w'][i]), int(boxes['h'][i]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        bbox = (pt1[0], pt1[1], pt2[0], pt2[1])
        bbox = tuple(x * scale_multiplier for x in bbox)

        score = boxes['class_confidence'][i]
        class_id = boxes['class_id'][i]
        class_name = labelmap[class_id % len(labelmap)]
        bbox_txt = class_name
        if add_score:
            bbox_txt += f' {score:.2f}'
        color_tuple_rgb = classid2colors[class_id]
        img = bbv.draw_rectangle(img, bbox, bbox_color=color_tuple_rgb)
        img = bbv.add_label(img, bbox_txt, bbox, text_bg_color=color_tuple_rgb, top=True)

    return img


def draw_bboxes(img, boxes, labelmap=LABELMAP_GEN1) -> None:
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    for i in range(boxes.shape[0]):
        pt1 = (int(boxes['x'][i]), int(boxes['y'][i]))
        size = (int(boxes['w'][i]), int(boxes['h'][i]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        score = boxes['class_confidence'][i]
        class_id = boxes['class_id'][i]
        class_name = labelmap[class_id % len(labelmap)]
        color = colors[class_id * 60 % 255]
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(img, pt1, pt2, color, 1)
        cv2.putText(img, class_name, (center[0], pt2[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        cv2.putText(img, str(score), (center[0], pt1[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

def visualize(video_writer: cv2.VideoWriter, ev_tensors: torch.Tensor, labels_yolox: torch.Tensor, pred_processed: torch.Tensor, dataset_name:str):
    img = ev_repr_to_img(ev_tensors.squeeze().cpu().numpy())
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if labels_yolox is not None:
        img = draw_bboxes_bbv(img, labels_yolox, dataset_name)

    if pred_processed is not None:
        img = draw_bboxes_bbv(img, pred_processed, dataset_name)

    video_writer.write(img)

def create_video(data: pl.LightningDataModule , model: pl.LightningModule, show_gt: bool, show_pred: bool, output_path: str, fps: int, num_sequence: int, dataset_mode: DatasetMode):  

    data_size =  dataset2size[data.dataset_name]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, data_size)


    if dataset_mode == "train":
        data.setup('fit')
        data_loader = data.train_dataloader()
        model.setup("fit")
    elif dataset_mode == "val":
        data.setup('validate')
        data_loader = data.val_dataloader()
        model.setup("validate")
    elif dataset_mode == "test":
        data.setup('test')
        data_loader = data.test_dataloader()
        model.setup("test")
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")
    

    num_classes = len(dataset2labelmap[data.dataset_name])

    ## device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if show_pred:
        model.eval()
        model.to(device)  # モデルをデバイスに移動
        rnn_state = RNNStates()
        prev_states = rnn_state.get_states(worker_id=0)
        size = model.in_res_hw
        input_padder = InputPadderFromShape(size)

    sequence_count = 0
    

    for batch in tqdm(data_loader):
        data_batch = batch["data"]

        ev_repr = data_batch[DataType.EV_REPR]
        labels = data_batch[DataType.OBJLABELS_SEQ]
        is_first_sample = data_batch[DataType.IS_FIRST_SAMPLE]

        if is_first_sample.any():
            sequence_count += 1
            if sequence_count > num_sequence:
                break

        labels_yolox = None
        pred_processed = None

        sequence_len = len(ev_repr)
        for tidx in range(sequence_len):
            ev_tensors = ev_repr[tidx]
            ev_tensors = ev_tensors.to(torch.float32).to(device)  # デバイスに移動

            ##ラベルを取得
            if show_gt:
                current_labels, valid_batch_indices = labels[tidx].get_valid_labels_and_batch_indices()
                if len(current_labels) > 0:
                    labels_yolox = ObjectLabels.get_labels_as_batched_tensor(obj_label_list=current_labels, format_='yolox')

            ## モデルの推論
            if show_pred:
                ev_tensors_padded = input_padder.pad_tensor_ev_repr(ev_tensors)
                if model.mdl.model_type == 'DNN':
                    predictions, _ = model(ev_tensors_padded)
                elif model.mdl.model_type == 'RNN':
                    predictions, _, states = model(x=ev_tensors_padded, previous_states=prev_states)
                    prev_states = states
                    rnn_state.save_states_and_detach(worker_id=0, states=prev_states)
                
                pred_processed = postprocess(prediction=predictions, num_classes=num_classes, conf_thre=0.1, nms_thre=0.45)

            ## 可視化
            visualize(video_writer, ev_tensors, labels_yolox, pred_processed, data.dataset_name)

    print(f"Video saved at {output_path}")
    video_writer.release()