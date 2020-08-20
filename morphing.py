import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltpath
import os
import subprocess

from multiprocessing import Pool, TimeoutError
from PIL import Image
from scipy.spatial import Delaunay
from scipy.spatial import voronoi_plot_2d, Voronoi
from subprocess import Popen, PIPE
# from threading import Thread

def specify_points(img):
    fig = plt.figure()
    fig.set_label('Select corresponding points in the image')
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    clicked = []

    def on_mouse_pressed(event):
        x = round(event.xdata)
        y = round(event.ydata)
        plt.plot(x, y, 'r+')
        clicked.append([x, y])

    fig.canvas.mpl_connect('button_press_event', on_mouse_pressed)

    return clicked


def get_points_in_triangulation(triangle, triangulation):
    '''
    Retrieves a list of points that are within the bounds of a triangle.

    Args:
      - triangle: The 3x2 matrix of points which are the vertices of the triangle.
      - triangulation: The scipy.spatial.Delaunay triangulation.

    Returns:
      - Array of points
    '''
    xs = list()
    ys = list()
    for vertex in triangle[triangulation.simplices][0]:
        xs.append(vertex[0])
        ys.append(vertex[1])
    points = list()
    for x in range(int(min(xs)), int(max(xs)) + 1):
        for y in range(int(min(ys)), int(max(ys)) + 1):
            simplices = triangulation.find_simplex(np.array([(x, y)]))
            if simplices[0] != -1:
                points.append([x, y])
    return np.array(points)


def compute_frame(triangulation, source_points, target_points, t, shape):
    '''
    Computes a frame of the image morph.

    Args:
      - triangulation: The scipy.spatial.Delaunay triangulation.
      - source_points: The list of selected points in the source image
        that were used to determine the triangulation.
      - target_points: The list of selected points in the target image
        that correspond to the selected points in the source image.
      - t: The time value in the range [0,1]
      - shape: The shape of the frame which should match the shape of
        the original source and target images.

    Returns:
      - The frame of the morphing at time t.
    '''
    frame = np.zeros(shape=shape, dtype='uint8')

    # The number of triangles is determined by the simplices attribute
    num_triangles = len(triangulation.simplices)
    average_triangles = np.zeros(shape=(num_triangles, 3, 2), dtype=np.float32)

    for triangle_index in range(0, num_triangles):
        simplices = triangulation.simplices[triangle_index]
        for v in range(0, 3):
            simplex = triangulation.simplices[triangle_index][v]
            P = source_points[simplex]
            Q = target_points[simplex]
            average_triangles[triangle_index][v] = P + t * (Q - P)

        # Compute the affine projection to the source and target triangles
        source_triangle = np.float32([
            source_points[simplices[0]],
            source_points[simplices[1]],
            source_points[simplices[2]]
        ])
        target_triangle = np.float32([
            target_points[simplices[0]],
            target_points[simplices[1]],
            target_points[simplices[2]]
        ])
        average_triangle = np.float32(average_triangles[triangle_index])
        source_transform = cv2.getAffineTransform(average_triangle, source_triangle)
        target_transform = cv2.getAffineTransform(average_triangle, target_triangle)

        average_triangulation = Delaunay(average_triangle)

        # For each point in the average triangle, find the corresponding points
        # in the source and target triangle, and find the weighted average.
        average_points = get_points_in_triangulation(average_triangle, average_triangulation)
        for point in average_points:
            source_point = np.transpose(np.dot(source_transform, np.transpose(np.array([point[0], point[1], 1]))))
            target_point = np.transpose(np.dot(target_transform, np.transpose(np.array([point[0], point[1], 1]))))

            # Perform a weighted average per-channel
            for c in range(0, shape[2]):
                source_val = source_img[int(source_point[1]), int(source_point[0]), c]
                target_val = target_img[int(target_point[1]), int(target_point[0]), c]
                frame[point[1], point[0], c] = round((1 - t) * source_val + t * target_val)
    return frame


def write_frame(frame, frame_number, group):
    os.makedirs('./images/outputs/{}/'.format(group), exist_ok=True)
    cv2.imwrite('./images/outputs/{}/frame_{}.png'.format(group, str(frame_number)), frame)

def process_func(triangulation, source_points, target_points, t, frame_num, shape, group_name):
    frame = compute_frame(triangulation, source_points, target_points, t, shape)
    write_frame(frame, frame_num, group_name)


source_img = cv2.imread('outputs/Gyeongbokgung/1.png')
target_img = cv2.imread('outputs/Gyeongbokgung/2.png')

assert source_img.shape == target_img.shape
H, W, C = source_img.shape

# fix, axes = plt.subplots(1, 2)
# [a.axis('off') for a in axes.ravel()]
# axes[0].imshow(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
# axes[1].imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))

source_points = specify_points(source_img)
target_points = specify_points(target_img)

triangulation = Delaunay(source_points)

duration = 3 # in seconds
fps = 10
num_frames = duration * fps

pool = Pool(processes=4)
results = []
for frame_num in range(0, num_frames):
    t = frame_num / num_frames
    res = pool.apply_async(process_func, (triangulation, source_points, target_points, t, frame_num, (H, W, C), 'test'))
    results.append(res)

for res in results:
    res.get(timeout=None)

def create_gif(group):
    '''
    Creates an animated GIF from the frame files using ImageMagick.
    '''
    frames = glob.glob('./images/outputs/{}/*.png'.format(group))
    list.sort(frames, key=lambda x: int(x.split('_')[1].split('.png')[0]))
#     print(frames)
    with open('./images/outputs/{}/frame_list.txt'.format(group), 'w') as f:
        for frame in frames:
            f.write('{}\n'.format(frame))
    process = subprocess.Popen(['convert', '@./images/outputs/{}/frame_list.txt'.format(group), './images/outputs/{}/{}.gif'.format(group, group)], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate();
    if process.returncode != 0:
        print(stderr.decode('utf-8'))
#     os.system('convert @frame_list.txt {}.gif'.format(group))

gif = create_gif('test')



##################### error detected