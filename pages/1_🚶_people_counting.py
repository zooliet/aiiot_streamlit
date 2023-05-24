import streamlit as st

import os
import time

from ultralytics import YOLO
import torch
import supervision as sv
from supervision.draw.color import Color
import cv2

# st.set_page_config(layout="wide")

LINE_START = sv.Point(720, 330)
LINE_END = sv.Point(0, 330)

st.title("보행자 숫자")
st.markdown("---")

cfg_model_path = 'models/yolov8n-seg.pt'
# check if model file is available
if not os.path.isfile(cfg_model_path):
    st.warning("Model file not available!!!, please added to the model folder.", icon="⚠️")

# load model
model = YOLO(cfg_model_path)

# line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
# line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

box_annotator = sv.BoxAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=0.5,
    color=Color.white()
)

# mask_annotator = sv.MaskAnnotator(
#     # color=Color.red()
# )


vid_file = "videos/people_walking.avi"
cap = cv2.VideoCapture(vid_file)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = 0
output = st.empty()

count_text = st.markdown(0)
# st1_text = st.markdown(f"{height}")
# st1, st2, st3 = st.columns(3)
# with st1:
#     st.markdown("## 높이")
#     st1_text = st.markdown(f"{height}")
# with st2:
#     st.markdown("## 폭")
#     st2_text = st.markdown(f"{width}")
# with st3:
#     st.markdown("## FPS")
#     st3_text = st.markdown(f"{fps}")
#

prev_time = 0
curr_time = 0

for result in model.track(source=vid_file, show=False, stream=True, agnostic_nms=False, verbose=False):
    frame = result.orig_img
    # frame = cv2.resize(frame, (width, height))

    detections = sv.Detections.from_yolov8(result)
    detections = detections[detections.class_id == 0] 

    if result.boxes.id is not None:
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

    labels = [
        f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id 
        in detections
    ]

    count = len(labels)
    
    frame = box_annotator.annotate(
        scene=frame, 
        detections=detections,
        labels=labels,
        skip_label=True
    )

    # frame = mask_annotator.annotate(
    #     scene=frame, 
    #     detections=detections,
    #     opacity=0.5,
    # )

    # line_counter.trigger(detections)
    # line_annotator.annotate(frame=frame, line_counter=line_counter)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    count_text.markdown(f"**현재 {count} 명이 있습니다**")

    # st1_text.markdown(f"**{height}**")
    # st2_text.markdown(f"**{width}**")
    # st3_text.markdown(f"**{fps:.2f}**")

    output.image(frame)
    cv2.waitKey(30)
    # if (cv2.waitKey(30) == 27):
    #     break

cap.release()
st.button("Re-run")


