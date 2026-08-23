import numpy as np
import cv2
import pygsp.plotting
from pygsp.graphs import Graph, Grid2d
from matplotlib import pyplot as plt

logo_img = cv2.imread('Logo_telecos_2018.png', flags=cv2.IMREAD_GRAYSCALE)
ret,thresh_img = cv2.threshold(logo_img,110,255,cv2.THRESH_BINARY_INV)
cv2.imshow("Thresholded Image", thresh_img)
cv2.waitKey(0)
coords = cv2.findNonZero(thresh_img)
graph = Grid2d(20)
graph_coords = graph.coords
pygsp.plotting.plot_graph(graph)
plt.show(block=True)