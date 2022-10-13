import av.datasets
import matplotlib.pyplot
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
import os
import numpy as np


def task1(container):
    count = container.streams.video[0].frames
    w = container.streams.video[0].codec_context.coded_width
    h = container.streams.video[0].codec_context.coded_height
    y = np.zeros([count, w * h], dtype=float)
    counter = 0
    for frame in container.decode(0):
        rgb = np.asarray(frame.to_rgb().to_image())
        temp = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        y[counter, :] = np.reshape(temp, [w * h])
        counter += 1
    y_rez = np.reshape(y, [y.size])
    K = w * h
    R = []
    for shift_i in range(count):
        temp1 = y_rez[shift_i * K:]
        temp2 = y_rez[:y_rez.size - shift_i * K]
        r = 0.0
        M1 = np.mean(temp1)
        M2 = np.mean(temp2)
        for j in range(temp1.size):
            r = r + (temp1[j] - M1) * (temp2[j] - M2)
        R.append(r / (np.std(temp1) * np.std(temp2) * temp1.size))
        # print(r / (np.std(temp1) * np.std(temp2) * temp1.size))
    return R


def task2(container):
    output = av.open("test4.avi", "w")
    input_stream = container.streams.video[0]
    output_stream = output.add_stream(template=input_stream)
    l = list(container.demux(input_stream))
    l.reverse()
    for i, packet in enumerate(l):
        packet.pts = i
        packet.dts = i
        packet.stream = output_stream
        output.mux(packet)
    output.close()
    # F = []
    # for frame in container.decode(0):
    #     F.append(frame.to_rgb())
    # total_frames = len(F)
    # container_rez = av.open("test.mp4", mod="w")
    # stream = container.add_stream("mpeg4", rate=1)
    # stream.width = container.streams.video[0].codec_context.coded_width
    # stream.height = container.streams.video[0].codec_context.coded_height
    # stream.pix_fmt = "yuv420p"
    #


def task3(container1, container2):
    output = av.open("task4.avi", "w")
    input_stream1 = container1.streams.video[0]
    input_stream2 = container2.streams.video[0]
    output_stream = output.add_stream(template=input_stream1)
    l1 = list(container1.demux(input_stream1))
    l2 = list(container2.demux(input_stream2))
    l = l1 + l2
    for i, packet in enumerate(l):
        packet.pts = i
        packet.dts = i
        packet.stream = output_stream
        output.mux(packet)
    output.close()

container = []
container.append(av.open('./video/lr1_1.AVI'))
container.append(av.open('./video/lr1_2.AVI'))
container.append(av.open('./video/lr1_3.AVI'))
# t1_1 = task1(container[0])
# t1_2 = task1(container[1])
# t1_3 = task1(container[2])
# t1 = t1_1.copy()
# t1.reverse()
# t2 = t1_2.copy()
# t2.reverse()
# t3 = t1_3.copy()
# t3.reverse()
# plt.plot(range(-len(t1_1),0), t1, 'r')
# plt.plot(range(len(t1_1)), t1_1, 'r')
# plt.plot(range(-len(t1_2),0), t2, 'g')
# plt.plot(range(len(t1_2)), t1_2, 'g')
# plt.plot(range(-len(t1_3),0), t3, 'b')
# plt.plot(range(len(t1_3)), t1_3, 'b')
# plt.show()

# task2(container[0])

# task3(container[0], container[1])



container[0].close()
container[1].close()
container[2].close()
