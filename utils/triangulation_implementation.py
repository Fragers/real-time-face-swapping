from utils.model_implementation import *

import cv2
import json


def readPoints(path):
    # Create an array of points.
    new_points = []

    # Read points
    with open(path, 'r') as f:
        points = json.load(f)
    for point in points:
        new_points.append(tuple(point))

    return new_points


# Находим матрицу афинного преобразование по координатам и применяем его на исходный треугольник
def applyAffineTransform(src, srcTri, dstTri, size):
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


# calculate delanauy triangle
def calculateDelaunayTriangles(rect, points, points_dict):
    # create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList()

    delaunayTri = []

    pt = []

    for index, t in enumerate(triangleList):
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []

            ind.append(points_dict[pt1])
            ind.append(points_dict[pt2])
            ind.append(points_dict[pt3])

            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []

    return delaunayTri


# Warps and alpha blends triangular regions from img1 and img2 to img
# img1 - исходное изображение, img2 - пустое изображение, в которое поместим обновленные треугольника
# t1 - координаты исзодного треульника, t2 - координаты текущего треульника
def warpTriangle(img1, img2, t1, t2):
    # Создаем прямоугольники по размеру окружаюего прямоугольника для каждого треульника
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    # Создаем массивы, содержащие координаты треугольников, сдвинутые, в начало координат
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Создаем пустую маску по координатам текущего треульноника
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Получаем кусочек фотографии, ограниченный прямоугольником для треугольника с исходного фото
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])
    # Применяем аффинное преобразование по координатам текущего треугольника и исходного
    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    # Применям маску на прямоугольник
    img2Rect = img2Rect * mask

    # Помещаем обновленный треугольник на пустую фотографию
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
            (1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect
    # img2Rect = cv2.resize(img2Rect, (200, 200))
    # cv2.imshow("123", img2Rect)
    # return img2
