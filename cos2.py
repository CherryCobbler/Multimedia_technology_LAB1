import av.datasets
import matplotlib.pyplot as plt
import numpy as np


def metric_SAD(block1, block2):
    difference = block1 - block2
    abs_difference = np.abs(difference)
    sum_value = sum(sum(abs_difference))
    return sum_value


def get_blocks(frame, R, x, y, w, h):
    list_block = []
    if y - R >= 0:  # top
        if x - R >= 0:
            if len(frame[x - R:x, y - R:y]) != 0:
                list_block.append([x - R, y - R, frame[x - R:x, y - R:y], 1])  # 1
        if x + R <= w:
            if len(frame[x:x + R, y - R:y])!=0:
                list_block.append([x, y - R, frame[x:x + R, y - R:y], 2])  # 2
        if x + 2 * R <= w:
            if len(frame[x + R:x + 2 * R, y - R:y]) != 0:
                list_block.append([x + R, y - R, frame[x + R:x + 2 * R, y - R:y], 3])  # 3
    if y + R <= h:  # medium
        if x - R >= 0:
            if len(frame[x - R:x, y:y + R]) != 0:
                list_block.append([x - R, y, frame[x - R:x, y:y + R], 4])  # 4
        if x + 2 * R <= w:
            if len(frame[x + R:x + 2 * R, y:y + R]) != 0:
                list_block.append([x + R, y, frame[x + R:x + 2 * R, y:y + R], 5])  # 5
    if y + 2 * R <= h:  # down
        if x - R >= 0:
            if len(frame[x - R:x, y + R:y + 2 * R])!=0:
                list_block.append([x - R, y + R, frame[x - R:x, y + R:y + 2 * R], 6])  # 6
        if x + R <= w:
            if len(frame[x:x + R, y + R:y + 2 * R]) !=0:
                list_block.append([x, y + R, frame[x:x + R, y + R:y + 2 * R], 7])  # 7
        if x + 2 * R <= w:
            if len(frame[x + R:x + 2 * R, y + R:y + 2 * R])!=0:
                list_block.append([x + R, y + R, frame[x + R:x + 2 * R, y + R:y + 2 * R], 8])  # 8
    return list_block


def search_3step(frame, first_frame, R, w, h):
    list_list_vec = []
    for x in range(0, w - R, R):
        for y in range(0, h - R, R):
            r = R
            x_temp = x
            y_temp = y
            list_vec = []
            while r != 1:
                block_current = frame[x_temp:x_temp + r, y_temp:y_temp + r]
                list_block = get_blocks(first_frame, r, x_temp, y_temp, w, h)
                SAD = metric_SAD(list_block[0][2], block_current)
                SAD_i = 0
                for i in range(len(list_block)):
                    temp = metric_SAD(list_block[i][2], block_current)
                    if SAD > temp:
                        SAD_i = i
                        SAD = temp
                list_vec.append([list_block[SAD_i][0], list_block[SAD_i][1]])
                x_temp = list_block[SAD_i][0]
                y_temp = list_block[SAD_i][1]
                r = r // 2
            list_list_vec.append(list_vec)
    return list_list_vec


def motion_vectors(container, count, R):
    if count <= 0:
        count = container.streams.video[0].frames
    count += 1
    frame_list = []
    for frame in container.decode(0):
        if count == 0:
            break
        rgb = np.asarray(frame.to_rgb().to_image())
        temp = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        frame_list.append(temp)
        count -= 1
    frame_list.pop(0)
    frame_first = frame_list.pop(0)
    w = container.streams.video[0].codec_context.coded_width - 1
    h = container.streams.video[0].codec_context.coded_height - 1
    # беру фрейм, беру центральный блок, беру набор всех этих блоков с 8 в округе
    for frame in frame_list:
        temp = search_3step(frame,frame_first,R,w,h)
        print(temp)
        print()

container = av.open('./video/lr1_1.AVI')
motion_vectors(container, 8, 8)
container.close()
