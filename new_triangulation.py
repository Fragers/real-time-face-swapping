# Read points from text file
from utils.triangulation_implementation import *
import math
import time


def getLength(point1, point2, pointNose):
    l1 = math.sqrt(float(point1[0] - pointNose[0]) ** 2 + float(point1[1] - pointNose[1]) ** 2)
    l2 = math.sqrt(float(point2[0] - pointNose[0]) ** 2 + float(point2[1] - pointNose[1]) ** 2)
    return (l1 + l2) / 2


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('settings.ini')
    filename1 = config["Settings"]["filepath"]
    backname = config["Settings"]["backpath"]
    sizeW = int(config["Settings"]["sizeW"])
    sizeH = int(config["Settings"]["sizeH"])
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 24)  # Частота кадров
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, round(sizeW * 1))  # Ширина кадров в видеопотоке.
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, round(sizeH * 1))
    handler = Handler('new_model/2d106det', 0, ctx_id=-1,
                      det_size=224)  # чем меньше размер картинки тем быстрее инференс, но точность ниже, норм при 120..


    # Make sure OpenCV is version 3.0 or above
    # (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # Read images

    print(filename1)
    prev_frame = None
    img1 = cv2.imread(filename1)
    img1 = cv2.resize(img1, (sizeW, sizeH))
    img1_1 = img1.copy()
    img1_1 = cv2.resize(img1_1, (sizeW - 20, sizeH - 20))

    backfile = cv2.imread(backname)
    backfile = cv2.resize(backfile, (sizeW, sizeH))
    if config["Settings"]["landmarks_type"] == "from_the_file":
        points1 = readPoints(config["Settings"]["landmarks_source"])
    else:
        preds_source = handler.get(img1, get_all=False)

        # print(img1.shape)
        points1 = []
        for pred in preds_source:
            pred = np.round(pred).astype(np.int)
            for i in range(pred.shape[0]):
                p = tuple(pred[i])
                points1.append(p)
    #
    # hull1 = []
    # for i in range(len(points1)):
    #     hull1.append(points1[i])
    nose_point1 = points1[80]
    start_size = getLength(points1[1], points1[17], nose_point1)
    start_time = time.time()
    with pyvirtualcam.Camera(width=sizeW, height=sizeH, fps=15, print_fps=True) as cam:
        while True:
            _, img2 = cap.read()
            img_copy = img2.copy()
            # img1Warped = np.copy(img2)
            img1Warped = np.zeros_like(img2)
            # img_copy = img2.copy() #see landmarks
            # Read array of corresponding points
            # points1 = readPoints(filename1 + '.txt')
            # points2 = readPoints(filename2 + '.txt')

            points2 = []
            # start_time = time.time()
            preds_target = handler.get(img2, get_all=False)
            # print(time.time() - start_time)

            color = (200, 160, 75)

            for pred in preds_target:
                pred = np.round(pred).astype(np.int)
                for i in range(pred.shape[0]):
                    p = tuple(pred[i])
                    # cv2.circle(img_copy, tuple(pred[i]), radius=1, color=color)
                    # print(pred[i])
                    points2.append(p)

            # Find convex hull
            hull1 = []
            hull2 = []

            try:
                convHullIndex = cv2.convexHull(np.array(points2), returnPoints=False)
                hullIndex = [i for i in range(len(points2))]
            except:
                if prev_frame is not None:
                    cv2.imshow('video', output)
                    # tim = cv2.cvtColor(output, cv2.COLOR_RGB2RGBA)
                    cam.send(tim)
                    continue
                else:
                    continue
            # ненужный цикл
            if len(hullIndex) != 106:
                continue
            for i in range(0, len(hullIndex)):
                hull1.append(points1[int(hullIndex[i])])
                hull2.append(points2[int(hullIndex[i])])

            convexHull = []

            for i in range(0, len(convHullIndex)):
                convexHull.append(points2[int(convHullIndex[i])])
            # print(len(hullIndex))

            # if prev_frame is not None:
            #     cv2.imshow('video', output)
            #     # tim = cv2.cvtColor(output, cv2.COLOR_RGB2RGBA)
            #     cam.send(tim)
            #     continue
            # else:
            #     continue
            # Find delanauy traingulation for convex hull points

            sizeImg2 = img2.shape
            # print(sizeImg2)
            rect = (0, 0, sizeW, sizeH)
            points_dict = dict()
            mxx = 0
            mxy = 0
            for index, curPoint in enumerate(points2):
                # print(curPoint)
                points_dict[curPoint] = index
            try:
                dt = calculateDelaunayTriangles(rect, hull2, points_dict)
            except:
                continue
            if len(dt) == 0:
                quit()

            # Apply affine transformation to Delaunay triangles
            for i in range(0, len(dt)):
                t1 = []
                t2 = []

                # get points for img1, img2 corresponding to the triangles
                for j in range(0, 3):
                    t1.append(hull1[dt[i][j]])
                    t2.append(hull2[dt[i][j]])

                # img1Warped = warpTriangle(img1, img1Warped.copy(), t1, t2)
                warpTriangle(img1, img1Warped, t1, t2)

            nosePoint = points2[80]
            new_size = getLength(points2[1], points2[17], nosePoint) + 20
            # making mask for
            hull8U = []
            for i in range(0, len(convexHull)):
                hull8U.append((convexHull[i][0], convexHull[i][1]))

            mask = np.zeros(img2.shape, dtype=img2.dtype)

            cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

            r = cv2.boundingRect(np.float32([hull2]))

            # print(start_size, new_size)
            center = (
                r[0] - nosePoint[0] + sizeW // 2 + int(r[2] / 2), r[1] - nosePoint[1] + sizeH // 2 + int(r[3] / 2))

            old_output = np.uint8(img1Warped)

            T = np.float32([[1, 0, -nosePoint[0] + sizeW // 2], [0, 1, -nosePoint[1] + sizeH // 2]]) #centering
            M = cv2.getRotationMatrix2D(center, 0, start_size / new_size) #scaling
            M = np.array(M)
            T = np.array(T)

            output = cv2.warpAffine(old_output, T, (sizeW, sizeH))
            output = cv2.warpAffine(output, M, (sizeW, sizeH))

            mask = cv2.warpAffine(mask, T, (sizeW, sizeH))
            mask = cv2.warpAffine(mask, M, (sizeW, sizeH))

            new_output = cv2.seamlessClone(output, backfile, mask, center, cv2.NORMAL_CLONE)
            tim = cv2.cvtColor(new_output, cv2.COLOR_BGR2RGBA)

            # centering
            # output = np.uint8(img1Warped)

            # tim = cv2.cvtColor(output, cv2.COLOR_BGR2RGBA)
            # nosePoint = points2[53]
            # T = np.float32([[1, 0, -nosePoint[0] + sizeW // 2], [0, 1, -nosePoint[1] + sizeH // 2]])
            # new_output1 = cv2.warpAffine(output, T, (sizeW, sizeH))
            #
            # cv2.imshow('video1', new_output1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (0, 0, 255)
            thickness = 2
            # cv2.putText(output, "fps: " + str(round(1 / (time.time() - start_time), 2)), (30, 30), font, fontScale,
            #             color, thickness, cv2.LINE_AA)
            # cv2.putText(output, "inference: " + str(round(1 / (time.time() - start_time), 2)), (30, 60), font, fontScale,
            #             color, thickness, cv2.LINE_AA)

            start_time = time.time()
            cv2.imshow('video', new_output)
            # cv2.imshow('video1', img_copy)

            # cv2.imshow('video1', img_copy)

            prev_frame = True

            # output = tim
            cam.send(tim)
            cv2.waitKey(1)
            # cam.sleep_until_next_frame()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # cv2.imshow("Face Swapped", output)
            # cv2.waitKey(0)

    cv2.destroyAllWindows()
