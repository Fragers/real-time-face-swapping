# Read points from text file
from utils.triangulation_implementation import *
import math


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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, sizeW)  # Ширина кадров в видеопотоке.
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, sizeH)
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
    img1_1 = cv2.resize(img1_1, (sizeW-20, sizeH-20))

    backfile = cv2.imread(backname)
    backfile = cv2.resize(backfile, (sizeW, sizeH))

    preds_source = handler.get(img1, get_all=False)

    print(img1.shape)
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
    with pyvirtualcam.Camera(width=sizeW, height=sizeH, fps=15, print_fps=True) as cam:
        while True:
            _, img2 = cap.read()
            # img1Warped = np.copy(img2)
            img1Warped = np.zeros_like(img2)
            # img_copy = img2.copy() #see landmarks
            # Read array of corresponding points
            # points1 = readPoints(filename1 + '.txt')
            # points2 = readPoints(filename2 + '.txt')

            points2 = []

            preds_target = handler.get(img2, get_all=False)

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
            if len(hullIndex) == 0:
                continue

            nosePoint = points2[80]
            new_size = getLength(points2[1], points2[17], nosePoint) + 20
            points2 = np.array(points2)
            r = cv2.boundingRect(np.float32([points2]))

            # print(start_size, new_size)
            center = (r[0] - nosePoint[0] + sizeW // 2 + int(r[2] / 2), r[1] - nosePoint[1] + sizeH // 2 + int(r[3] / 2))

            img2 = np.uint8(img2)
            T = np.float32([[1, 0, -nosePoint[0] + sizeW // 2], [0, 1, -nosePoint[1] + sizeH // 2]])
            M = cv2.getRotationMatrix2D(center, 0, start_size / new_size)
            M = np.array(M)
            T = np.array(T)
            img2 = cv2.warpAffine(img2, T, (sizeW, sizeH))
            img2 = cv2.warpAffine(img2, M, (sizeW, sizeH))
            points2 = cv2.warpAffine(points2, T, (sizeW, sizeH))
            points2 = cv2.warpAffine(points2, M, (sizeW, sizeH))

            convHullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

            for i in range(0, len(hullIndex)):
                hull1.append(points1[int(hullIndex[i])])
                hull2.append(points2[int(hullIndex[i])])

            convexHull = []

            for i in range(0, len(convHullIndex)):
                convexHull.append(points2[int(convHullIndex[i])])

            # Find delanauy traingulation for convex hull points
            sizeImg2 = img2.shape
            rect = (0, 0, sizeImg2[1], sizeImg2[0])
            points_dict = dict()
            for index, curPoint in enumerate(points2):
                # print(curPoint)
                points_dict[curPoint] = index

            dt = calculateDelaunayTriangles(rect, hull2, points_dict)

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




            #making mask for
            hull8U = []
            for i in range(0, len(convexHull)):
                hull8U.append((convexHull[i][0], convexHull[i][1]))

            mask = np.zeros(img2.shape, dtype=img2.dtype)

            cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))


            # T.transpose()
            # M = M * T
            # print(M)
            # break
            # M = M.transpose()

            mask = cv2.warpAffine(mask, T, (sizeW, sizeH))

            mask = cv2.warpAffine(mask, M, (sizeW, sizeH))

            # new_output = cv2.seamlessClone(output, img1, mask, center, cv2.NORMAL_CLONE)
            tim = cv2.cvtColor(output, cv2.COLOR_BGR2RGBA)

            # centering
            # output = np.uint8(img1Warped)

            # tim = cv2.cvtColor(output, cv2.COLOR_BGR2RGBA)
            # nosePoint = points2[53]
            # T = np.float32([[1, 0, -nosePoint[0] + sizeW // 2], [0, 1, -nosePoint[1] + sizeH // 2]])
            # new_output1 = cv2.warpAffine(output, T, (sizeW, sizeH))
            #
            # cv2.imshow('video1', new_output1)
            cv2.imshow('video', img1Warped)
            # cv2.imshow('video1', img_copy)

            prev_frame = True

            # output = tim
            cam.send(tim)
            cv2.waitKey(1)
            cam.sleep_until_next_frame()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # cv2.imshow("Face Swapped", output)
            # cv2.waitKey(0)

    cv2.destroyAllWindows()
