import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
from annotated_text import annotated_text
import math

segmap = {
    "crack": [0, 0, 255],
    "cornerfracture": [0, 255, 0],
    "seambroken": [0, 255, 255],
    "patch": [255, 0, 0],
    "repair": [255, 0, 255],
    "slab": [255, 255, 0],
    "light": [255, 255, 255]
}


def increment_counter(files):
    st.session_state.index = (st.session_state.index + 1) % len(files)


def decrement_counter(files):
    st.session_state.index = (st.session_state.index - 1) % len(files)


def set_counter_slider():
    st.session_state.index = st.session_state.mannul_slider


def set_counter_input():
    st.session_state.index = st.session_state.mannul_input


def getDist_P2P(Point0, PointA):
    distance = math.pow((Point0[0] - PointA[0]), 2) + math.pow((Point0[1] - PointA[1]), 2)
    distance = math.sqrt(distance)
    return int(distance)


def LoadAnno(img, MASKFolder, files):
    with open(os.path.join(MASKFolder, files[st.session_state.index] + '.json'), 'r') as obj:
        dict = json.load(obj)
        for shape in dict['shapes']:
            if shape['label'] in segmap:
                if shape['shape_type'] == 'polygon':
                    pts = np.array(shape['points'], np.int32)
                    # cv2.polylines(img, [pts], True, segmap[shape['label']], 1)
                    cv2.fillPoly(img, [pts], segmap[shape['label']])
                elif shape['shape_type'] == 'circle':
                    pts = np.array(shape['points'], np.int32)
                    cv2.circle(img, pts[0], getDist_P2P(pts[0], pts[1]), segmap[shape['label']], -1)


def main():
    st.set_page_config(layout="wide")
    # st.sidebar.subheader('Choose Folder:')
    Folder = st.sidebar.text_input('Choose Folder:', '/home/vintinshaw/ViT-Adapter/segmentation/data/APD_10976/images')
    # st.sidebar.subheader('Choose splitFile:')
    splitFile = st.sidebar.text_input('Choose splitFile:',
                                      '/home/vintinshaw/ViT-Adapter/segmentation/data/APD_10976/splits/test.txt')
    files = []
    if os.path.isfile(splitFile) and os.path.exists(splitFile):
        with open(splitFile, 'r') as f:
            for line in f:
                files.append(line.strip('\n'))
    else:
        files = os.listdir(Folder)
        files = [os.path.splitext(i)[0] for i in files]
    # st.write(files)
    # st.sidebar.write(f'The current Folder has {len(files)} Pics:')
    # st.sidebar.subheader('Choose GT json folder:')
    GTFolder = st.sidebar.text_input('Choose GT json folder', '/dataRep3/APD/APD/json')
    # st.sidebar.subheader('Choose Pred json folder:')
    PREDFolder = st.sidebar.text_input('Choose Pred json folder', '/home/vintinshaw/ViT-Adapter/segmentation/json_dir')

    # loadpred=st.sidebar.checkbox("Pred",value=True)
    # loadgt=st.sidebar.checkbox("GT")
    # st.sidebar.subheader('Choose mask type:')
    loadAnno = st.sidebar.radio('Choose mask type:', ('None', 'Pred', 'GT'), index=2)
    if 'index' not in st.session_state:
        st.session_state['index'] = 0
    col1, col2, col3, = st.sidebar.columns([1, 2, 1])
    col1.button('prev', on_click=decrement_counter, args=(files,))
    col2.number_input('Mannel Index', value=st.session_state.index, min_value=0, max_value=len(files) - 1,
                      on_change=set_counter_input, key='mannul_input', label_visibility="collapsed")
    col3.button('next', on_click=increment_counter, args=(files,))
    st.sidebar.slider('Select Pic index:', value=st.session_state.index, min_value=0, max_value=len(files) - 1,
                      on_change=set_counter_slider, key='mannul_slider', label_visibility="collapsed")

    img = cv2.imread(os.path.join(Folder, files[st.session_state.index] + '.png'))

    if loadAnno == 'None':
        pass
    elif loadAnno == 'Pred':
        LoadAnno(img, PREDFolder, files)
    elif loadAnno == 'GT':
        LoadAnno(img, GTFolder, files)
    with st.sidebar:

        annotated_text(("crack", "裂缝", "rgb(0, 0, 255)"))
        annotated_text(("cornerfracture", "板角剥落", "rgb(0, 255, 0)"))
        annotated_text(("seambroken", "接缝破碎", "rgb(0, 255, 255)"))
        annotated_text(("patch", "补丁", "rgb(255, 0, 0)"))
        annotated_text(("repair", "修补", "rgb(255, 0, 255)"))
        annotated_text(("slab", "板缝", "rgb(255, 255, 0)"))
        annotated_text(("light", "灯", "rgb(255, 255, 255)"))

    # if loadpred and loadgt:
    #     Loadpred(img,MASKFolder,files)
    #     Loadgt(img,GTFolder,files)
    # elif loadgt:
    #     Loadgt(img,GTFolder,files)
    # elif loadpred:
    #     Loadpred(img,MASKFolder,files)
    # else:
    #     pass

    st.image(img, use_column_width=True)
    st.write(os.path.join(Folder, files[st.session_state.index] + '.png'))


if __name__ == '__main__':
    main()
