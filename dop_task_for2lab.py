import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
import av
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_img(img):
    picture = Image.fromarray(img.astype('uint8'))
    plt.imshow(picture)
    plt.show()


def PSNR(a, b, l=8):
    if a.shape != b.shape:
        return None

    error = np.sum((a - b) ** 2)
    res = a.shape[0] * a.shape[1] * (((2 ** l) - 1) ** 2) / error
    return 10 * np.log10(res)


def clip(img):
    img[img > 255] = 255
    img[img < 0] = 0
    return img


def getY(img):
    newimg = np.empty(img.shape, dtype='float')
    red = img[:, :, 2].astype('float')
    green = img[:, :, 1].astype('float')
    blue = img[:, :, 0].astype('float')
    newimg[:, :, 0] = 0.299 * red + 0.587 * green + 0.114 * blue  # y
    newimg[:, :, 1] = 0.299 * red + 0.587 * green + 0.114 * blue  # y
    newimg[:, :, 2] = 0.299 * red + 0.587 * green + 0.114 * blue  # y
    newimg = clip(newimg)
    return newimg


def SAD(block1, block2):
    tmp = np.abs(block1 - block2)
    res = sum(sum(tmp))
    return res


def mon_search(pic_ref, pic2, r, block_size):
    dx = block_size
    dy = block_size
    #
    vectors = []
    for x_cur in range(0, pic2.shape[0], dx):
        for y_cur in range(0, pic2.shape[1], dy):
            vectors.append([deep_search(x_cur, y_cur, pic2[:, :, 0], pic_ref[:, :, 0], dx, dy, r)])
    return vectors


def deep_search(x, y, pic2, pic_ref, dx, dy, r):
    r = 1
    minSAD = 9223372936854775807
    minx = x
    miny = y
    #print([x,y])
    block = pic2[x:x + dx, y:y + dy]
    for x_r in range(x - (r * dx), x + (r * dy), dx):
        for y_r in range(y - (r * dx), y + (r * dy), dy):
            if x_r > 0 and y_r > 0:
                try:
                    cur_sad = SAD(block, pic_ref[x_r:x_r + dx, y_r:y_r + dy])
                except ValueError:
                    continue
                if cur_sad < minSAD:
                    minSAD = cur_sad
                    mix = x_r - x
                    miny = y_r - y
    return minx, miny





def motion_compensation(ref, motion_vecs, block_size):
    res = np.zeros(ref.shape)
    x_size = block_size
    y_size = block_size
    for i, vec_row in enumerate(motion_vecs):
        # print(vec_row)
        for j, vec in enumerate(vec_row):
            x, y = i * x_size, j * y_size
            xr, yr = i * x_size + vec[0], j * y_size + vec[1]
            try:
                try:
                    res[x:x + x_size, y:y + y_size, :] = ref[xr:xr + x_size, yr:yr + y_size, :]
                except IndexError:
                    res[x:x + x_size, y:y + y_size] = ref[xr:xr + x_size, yr:yr + y_size]
            except ValueError:
                continue

    return res


def compilation(vectors, pic1, pic2, block_size):
    # print(vectors)
    comp_frame = motion_compensation(pic1, vectors, block_size)
    diff_frame = pic2 - comp_frame + 128
    diff_frame[diff_frame < 0] = 0
    diff_frame[diff_frame > 255] = 255
    newpic = diff_frame.astype('uint8')
    return newpic


def recovery(vectors, pic1, pic2, block_size):
    comp_frame = motion_compensation(pic1, vectors, block_size)
    diff_frame = comp_frame + pic2 - 128
    diff_frame[diff_frame < 0] = 0
    diff_frame[diff_frame > 255] = 255
    newpic = diff_frame  # .astype('uint8')
    return newpic


def init_q_chroma(R, s):
    res = np.empty(s)
    W, H = s[0], s[1]
    for i in range(W):
        for j in range(H):
            if len(s) == 3:
                res[i, j, :] = 1 + (i % 8 + j % 8) * R
            else:
                res[i, j] = 10 + (i % 8 + j % 8) * R
    return res


def quantize(G, m):
    return (G / m).round()  # .astype('uint8')


def dequantize(G, m):
    return (np.multiply(G, m))


def coder(img, qCHR):
    newpic = np.empty(img.shape)
    x = 0
    y = 0
    for x in range(0, img.shape[0], 8):
        for y in range(0, img.shape[1], 8):
            orig_block = dctn(img[x:x + 8, y:y + 8, 0], norm='ortho')
            temp = orig_block
            newpic[x:x + 8, y:y + 8, 0] = temp
            newpic[x:x + 8, y:y + 8, 1] = temp
            newpic[x:x + 8, y:y + 8, 2] = temp
    newpic = quantize(newpic, qCHR)
    return newpic


def decoder(img, qCHR):
    newpic = np.empty(img.shape)
    x = 0
    y = 0
    newpic = dequantize(img, qCHR)
    for x in range(0, img.shape[0], 8):
        for y in range(0, img.shape[1], 8):
            orig_block = img[x:x + 8, y:y + 8, 0]
            temp = idctn(orig_block, norm='ortho')
            newpic[x:x + 8, y:y + 8, 0] = temp
            newpic[x:x + 8, y:y + 8, 1] = temp
            newpic[x:x + 8, y:y + 8, 2] = temp
    return newpic


def save_img(img, path):
    image = Image.fromarray(img)
    image.save(path)


def run_length_encode(arr):
    last_nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i
    runs = []
    levels = []

    run = 0
    level = 0

    for i, elem in enumerate(arr):
        if i > last_nonzero + 1:
            runs.append(0)
            levels.append(0)
            break
        else:
            if elem == level:
                run += 1
            else:
                if run > 0:
                    runs.append(run)
                    levels.append(level)
                run = 1
                level = elem

    return runs, levels


def get_bc(x):
    res = np.abs(x)
    try:
        res[res == 0] = 1
    except TypeError:
        if res == 0:
            return 1
    return np.log2(res).astype('int') + 1


def get_pr(x):
    res = {}

    for i in x.flat:
        # print(i)
        try:
            res[i] += 1
        except KeyError:
            res[i] = 1

    for k in res.keys():
        res[k] /= x.size

    return res


def entropy1(x):
    Pe = get_pr(x)
    res = 0
    for i in range(len(Pe)):
        if (Pe[i] != 0):
            res += float(Pe[i]) * float(np.log2(float(Pe[i])))
    return int(-res) + 1


def entropy(x):
    r = 0
    f = get_pr(x)
    for i in f.values():
        if i == 0:
            continue
        r += i * np.log2(i)

    return -int(r) + 1


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def jpeg_processing(frame, vecs):
    b1, b2 = 0, 0

    x, y = frame.shape[0], frame.shape[1]
    zigzag = [  # (0, 0),
        (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4),
        (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0),
        (5, 1), (4, 2), (3, 3), (2, 4), (1, 5), (0, 6), (0, 7),
        (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0),
        (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6), (1, 7),
        (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (7, 3),
        (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6), (6, 5),
        (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)]

    last_dc = 0
    dc = []

    ac = 0

    for i in range(0, x, 8):
        for j in range(0, y, 8):
            block = frame[i:i + 8, j:j + 8]
            dc.append(block[0, 0] - last_dc)
            last_dc = block[0, 0]
            ac_s = [block[i] for i in zigzag]
            run, level = run_length_encode(ac_s)
            ac += (entropy(np.array(run)) + entropy(get_bc(level))) * len(run) + sum(get_bc(level))

    b1 = entropy(get_bc(dc)) * len(np.array(dc)) + sum(get_bc(dc)) + ac
    b2 = entropy(np.array(flatten(vecs))) * len(flatten(vecs))

    # print(vecs[0])
    # print(b1, b2)
    return b1 + b2


def p1(src, num, R):
    sstep = 33 #размеры области
    r = 8 #для локального поиска
    output = av.open('./video/lr1_4.AVI', "w")
    inputvid = av.open(src)
    frames = list(inputvid.decode(video=0))
    fr_count = len(frames)
    pic_ref = np.asarray(frames[1].to_rgb().to_image())
    tmp1 = getY(pic_ref)
    listimg = []
    qCHR = init_q_chroma(R, tmp1.shape)
    # num = fr_count
    it = tqdm(range(2, num))
    jpeg_size = 0
    my_size = 0
    stock_size = 0
    avg_psnrm = 0
    avg_psnrj = 0
    for i in it:
        tmp2 = np.asarray(frames[i].to_rgb().to_image())
        tmp2 = getY(tmp2)
        vectors = mon_search(tmp1, tmp2, r, sstep)
        img = compilation(vectors, tmp1, tmp2, sstep)
        save_img(img.astype('uint8'), f'media/diff_frames/{i}.bmp')
        imgc = coder(img, qCHR)  # diff_frames
        my_size += jpeg_processing(imgc[:, :, 0], vectors)
        imgd = decoder(imgc, qCHR)
        imgr = recovery(vectors, tmp1, imgd, sstep)
        avg_psnrm += PSNR(imgd[:, :, 0], img[:, :, 0])

        jpeg_pic1 = coder(tmp2, qCHR)
        jpeg_size += jpeg_processing(jpeg_pic1[:, :, 0], [])
        jpeg_pic = decoder(jpeg_pic1, qCHR)
        avg_psnrj += PSNR(tmp2[:, :, 0], jpeg_pic[:, :, 0])

        print(f'MPEGsize = {my_size}, JPEGsize = {jpeg_size}')
        print(f'PSNRm ={avg_psnrm / (i - 1)} PSNRj = {avg_psnrj / (i - 1)}')
        # avg_psnrm += cur_psnr
        # avg_psnrj += jpeg_psnr
        save_img(imgr.astype('uint8'), f'media/recover/{i}.bmp')
        tmp1 = tmp2
        listimg.append(imgr.astype('uint8'))
        stock_size += tmp2.shape[0] * tmp2.shape[1] * 8
    it.close()
    output_stream = output.add_stream('mpeg4')
    for i, pct in enumerate(listimg):
        frame = av.VideoFrame.from_ndarray(pct, format='rgb24')
        packet = output_stream.encode(frame)
        if packet is not None:
            output.mux(packet)
    inputvid.close()
    output.close()
    avg_psnrm = avg_psnrm / (num - 2)
    avg_psnrj = avg_psnrj / (num - 2)
    compress_my_factor = stock_size / my_size
    compress_jpeg_factor = stock_size / jpeg_size
    print(f'avg psnrm = {avg_psnrm}')
    print(f'avg psnrj = {avg_psnrj}')
    print(f'koef_comp_my = {compress_my_factor}')
    print(f'koef_comp_jpeg = {compress_jpeg_factor}')
    print('done')
    return avg_psnrm, avg_psnrj, compress_my_factor, compress_jpeg_factor


fig, axs = plt.subplots(3)
src = './video/lr1_2.AVI'
graph = []
graphj = []
compj = []
compm = []
num = 60
for R in range(0, num, 1):
    print(R)
    a, j, b, c = p1(src, num, R)
    graph.append(a)
    graphj.append(j)
    compj.append(c)
    compm.append(b)
axs[0].set_title(f'PSNR {src}')
axs[0].plot(graphj, label='jpeg')
axs[0].plot(graph, label='my')
axs[1].set_title(f'Compression Factor {src}')
axs[1].plot(compj, label='jpeg')
axs[1].plot(compm, label='my')
axs[2].set_title(f'Compression Factor to PSNR {src}')
axs[2].plot(graph, compj, label='jpeg')
axs[2].plot(graph, compm, label='my')
plt.legend()
plt.show()
