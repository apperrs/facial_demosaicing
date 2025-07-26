import json
import os
import random

import cv2
import numpy as np
from torch import optim

from torchsummary import summary
import onnxruntime as ort

from models.color_model import *
from matplotlib import pyplot as plt

batch_size = 18
epochs = 5

train_path = './mosaic/'
val_path = './mosaic_val/'

modified_path = '../../../FaceMosaicClean/CASIA-WebFace/'
val_modified_path = './val/'


def draw_mask(img, x1, y1, x2, y2, block_size=10):
    h, w, c = img.shape
    roi = img
    for i in range(y1, y2, block_size):
        for j in range(x1, x2, block_size):
            i_end = min(i + block_size, h)
            j_end = min(j + block_size, w)
            block = roi[i:i_end, j:j_end]
            block_blurred = cv2.blur(block, (block_size, block_size))
            roi[i:i_end, j:j_end] = block_blurred
    return img


def preprocess(path, path2):
    if not os.path.exists(path2 + 'A/'):
        os.makedirs(path2 + 'A/')
    if not os.path.exists(path2 + 'B/'):
        os.makedirs(path2 + 'B/')
    for dir in os.listdir(path):
        for file in os.listdir(path + dir + '/'):
            if 'json' in file:
                continue
            img = cv2.imread(path + dir + '/' + file)

            if not os.path.exists(path + '/' + dir + '/' + file + '.json'):
                continue
            dets = json.loads(open(path + '/' + dir + '/' + file + '.json', 'r').read())
            x1, y1, x2, y2 = 0, 0, 0, 0
            for det in dets:
                x1, y1, x2, y2 = map(int, det['xyxy'])
                det['xyxy'] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            a_img = draw_mask(img.copy(), x1, y1, x2, y2, block_size=random.randint(6, 19))
            a_img = a_img[y1:y2, x1:x2]

            b_img = img[y1:y2, x1:x2]
            a_img = cv2.resize(a_img, (256, 256))
            b_img = cv2.resize(b_img, (256, 256))
            cv2.imwrite(path2 + '/A/' + file.replace('.jpg', '.png'), a_img)
            cv2.imwrite(path2 + '/B/' + file.replace('.jpg', '.png'), b_img)


def draw_graph(train_loss, radio):
    plt.plot(range(0, len(train_losses) * radio, radio), train_loss, color='b', label="mse loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend(loc="best")
    plt.savefig('loss.png', dpi=400, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    preprocess(modified_path, train_path)
    preprocess(val_modified_path, val_path)

    session2 = ort.InferenceSession('cyclegan_G.onnx', providers=['CUDAExecutionProvider'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(123)
    model = DCEDN(6).to(device)

    summary(model, (6, 256, 256))

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    criterion = nn.MSELoss()

    length = len(os.listdir(train_path + 'A'))
    train_imgs_name = os.listdir(train_path + 'A')

    iter = 0
    train_losses = []

    for epoch in range(epochs):
        for i in range(0, length // batch_size * batch_size, batch_size):
            x = []
            y = []
            for j in range(batch_size):
                a_img = cv2.imread(train_path + 'A/' + train_imgs_name[i + j])
                a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)
                a_img = cv2.resize(a_img, [256, 256])
                a_img = a_img.transpose((2, 0, 1))

                input = (np.array([a_img]) / 127.5 - 1.).astype(np.float32)

                outputs = session2.run(None, {'input': input})
                out_img = ((np.array(outputs[0][0]).transpose((1, 2, 0)) + 1.) * 127.5).astype(np.uint8)
                out_img = out_img.transpose((2, 0, 1))
                x.append(np.concatenate([out_img / 127.5 - 1., a_img / 127.5 - 1.], axis=0))

                b_img = cv2.imread(train_path + 'B/' + train_imgs_name[i + j])
                b_img = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
                b_img = cv2.resize(b_img, [256, 256])
                b_img = b_img.transpose((2, 0, 1))

                b_img = (np.array(b_img) / 127.5 - 1.).astype(np.float32)
                y.append(b_img)

            x, y = np.array(x).astype(np.float32), np.array(y).astype(np.float32)
            x = torch.from_numpy(x).cuda()
            y = torch.from_numpy(y).cuda()
            outputs = model(x)

            loss = criterion(outputs, y)

            all_loss = loss

            print('Epoch:', '%04d' % (epoch + 1), 'Train loss =', '{:.6f}'.format(all_loss))

            if iter % 50 == 0:
                for n in os.listdir(val_path + 'A'):
                    v_img = cv2.imread(val_path + 'A/' + n)
                    v_img = cv2.cvtColor(v_img, cv2.COLOR_BGR2RGB)
                    v_img = cv2.resize(v_img, [256, 256])
                    v_img = v_img.transpose((2, 0, 1))
                    o_v_img = v_img.copy() / 127.5 - 1.

                    input = (np.array([v_img]) / 127.5 - 1.).astype(np.float32)

                    outputs = session2.run(None, {'input': input})
                    out_img = outputs[0][0]
                    cyc_img = (out_img.copy() + 1. )* 127.5
                    cyc_img = cyc_img.astype(np.uint8)
                    cyc_img = np.transpose(cyc_img, (1, 2, 0))
                    cv2.imwrite('./log/' + n + '_cyc_' + '.png', cv2.cvtColor(cyc_img, cv2.COLOR_RGB2BGR))

                    v_img = np.concatenate([out_img, o_v_img], axis=0)[np.newaxis, :, :, :].astype(np.float32)
                    v_img = torch.from_numpy(v_img).cuda()

                    y = model(v_img)

                    y = y.cpu().detach().numpy()[0]
                    y = y.transpose((1, 2, 0))
                    y = (y + 1.) * 127.5
                    y = y.astype(np.uint8)
                    img = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
                    cv2.imwrite('./log/'+ n + '_' +str(iter)+'.png', img)

                model.to(device)

            train_losses.append(all_loss.item())
            draw_graph(train_losses, 20)

            if iter >= 5000:
                torch.save(model.cpu(), './checkpoints/save.pt')
                model.to(device)
                exit(0)

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

            iter += 1


