import json
import numpy as np
import cv2
import argparse


R = 1
THICKNESS = 3
MARK_COLOR = (200, 160, 75)
CURRENT_MARK_COLOR = (230, 30, 25)
DELTA = 3

current_mark_idx = None


def draw_landmarks(img_name, img_src, marks_src, i_cur=None):
    img = img_src.copy()
    marks = np.round(marks_src).astype(np.int)
    for i in range(marks.shape[0]):
        point = tuple(marks[i])
        color = CURRENT_MARK_COLOR if i == i_cur else MARK_COLOR
        cv2.circle(img, point, R, color, THICKNESS, cv2.LINE_AA)

    cv2.imshow(img_name, img)


def find_near(marks, mouseX, mouseY, delta):
    for i, p in enumerate(marks):
        x, y = p
        if mouseX - delta <= x <= mouseX + delta and mouseY - delta <= y <= mouseY + delta:
            return i


def gen_cb(marks, delta):
    def on_click(event, x, y, p1, p2):
        global current_mark_idx
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if current_mark_idx:
            marks[current_mark_idx][:] = x, y
            current_mark_idx = None
        else:
            current_mark_idx = find_near(marks, x, y, delta)

    return on_click



def edit_landmarks(img_src, marks):
    window_name = 'image'
    draw_landmarks(window_name, img_src, marks, current_mark_idx)
    cv2.namedWindow(window_name)
    cb = gen_cb(marks, DELTA)
    cv2.setMouseCallback(window_name, cb)
    while True:
        draw_landmarks(window_name, img_src, marks, current_mark_idx)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break  # esc to quit

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img', help='image filename')
    parser.add_argument('src', default='src.json', help='file of landmarks array')
    parser.add_argument('--dst', default='result_landmarks.json', help='result landmarks file')
    args = parser.parse_args()

    with open(args.src) as f:
        arr = json.load(f)
    marks = np.array(arr, dtype=np.float32)
    img_src = cv2.imread(args.img)

    edit_landmarks(img_src, marks)

    with open(args.dst, 'w') as f:
        json.dump(marks.tolist(), f, indent=4)
