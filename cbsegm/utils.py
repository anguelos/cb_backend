import numpy as np
import json


def load_annotator_json_words(json_path:str):
    annotations = json.load(open(json_path, "r"))
    rectangles = annotations["rectangles_ltrb"]
    captions = annotations["captions"]
    assert len(rectangles)==len(captions)
    words=[]
    word_ltrb=[]
    for n in range(len(captions)):
        if captions[n].startswith("W@"):
            words.append(captions[n][2:])
            word_ltrb.append(rectangles[n])
    return np.array(word_ltrb), np.array(words)


def save_annotator_json_words(json_path:str, rectangles_ltrb, words=None):
    if words is None:
        captions = ["W@" for _ in range(rectangles_ltrb.shape[0])]
    elif isinstance(words, np.array):
        assert words.shape[0] == rectangles_ltrb.shape[0]
        captions = ["W@"+w for w in words.tolist()]
    elif isinstance(words, str):
        assert len(words) == rectangles_ltrb.shape[0]
        captions = ["W@"+w for w in words]
    annotations={}
    annotations["rectangles_ltrb"]=rectangles_ltrb.tolist()
    annotations["captions"]=captions
    with open(json_path, "w") as fd:
        json.dump(annotations, fd)


def render_bboxes(ltrb:np.array, image_size, bg_image=None, color=None, fill=True):
    left = ltrb[:, 0]
    top = ltrb[:, 1]
    right = ltrb[:, 2]
    bottom = ltrb[:, 3]
    if fill:
        integral = np.zeros((image_size[0] + 2, image_size[1] + 2))
        np.add.at(integral, (top, left), 1)
        np.add.at(integral, (bottom+1, right+1), 1)

        np.add.at(integral, (top, right+1), -1)
        np.add.at(integral, (bottom+1, left), -1)
        res = integral.cumsum(axis=0).cumsum(axis=1)
    else:
        h_integral = np.zeros((image_size[0] + 2, image_size[1] + 2))
        v_integral = np.zeros((image_size[0] + 2, image_size[1] + 2))

        np.add.at(h_integral, (top, left), 1)
        np.add.at(h_integral, (bottom+1, left), 1)
        np.add.at(h_integral, (top, right+1), -1)
        np.add.at(h_integral, (bottom+1, right+1), -1)

        np.add.at(v_integral, (top+1, left), 1)
        np.add.at(v_integral, (bottom, left), -1)
        np.add.at(v_integral, (top+1, right+1), 1)
        np.add.at(v_integral, (bottom, right+1), -1)

        res = h_integral.cumsum(axis=0)+v_integral.cumsum(axis=1)
    return res[:-2, :-2]






