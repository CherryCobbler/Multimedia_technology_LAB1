import av.datasets
import matplotlib.pyplot as plt
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


def dop(container, x, del_x, y, del_y):
    w = container.streams.video[0].codec_context.coded_width
    h = container.streams.video[0].codec_context.coded_height
    if x < 0 or del_x <= 0 or y < 0 or del_y <= 0 or x + del_x >= w or y + del_y >= h or x >= w or y >= h:
        print("Uncorrected data")
    else:
        list_frame = []
        for frame in container.decode(0):
            rgb = np.asarray(frame.to_rgb().to_image())
            temp = rgb[x:x + del_x, y:y + del_y]  # i love java but its too cool
            list_frame.append(av.VideoFrame.from_ndarray(temp, format=str('rgb24')))
        output_container = av.open('dop.avi', mode='w')
        output_stream = output_container.add_stream('h264', rate=25)
        out_packets = []
        for frame in list_frame:
            packet = output_stream.encode(frame)
            out_packets.append(packet)
        for packet in out_packets:
            if packet:
                output_container.mux(packet)
        output_container.close()


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
# plt.plot(range(-len(t1_1), 0), t1, 'r', label='avi1')
# plt.plot(range(len(t1_1)), t1_1, 'r')
# plt.plot(range(-len(t1_2), 0), t2, 'g', label='avi2')
# plt.plot(range(len(t1_2)), t1_2, 'g')
# plt.plot(range(-len(t1_3), 0), t3, 'b', label='avi3')
# plt.plot(range(len(t1_3)), t1_3, 'b')
# plt.legend()
# plt.show()
#
# task2(container[0])

task3(container[0], container[1])

dop(container[0], 100, 200, 100, 200)

container[0].close()
container[1].close()
container[2].close()
