import cv2
import numpy as np

import onnxruntime as ort

if __name__ == '__main__':
    session = ort.InferenceSession('cyclegan_G.onnx', providers=['CPUExecutionProvider'])
    session2 = ort.InferenceSession('dcedn.onnx', providers=['CPUExecutionProvider'])

    o_img = cv2.imread(input('Enter image path: '))
    o_img = cv2.resize(o_img, [256, 256])
    o_img = cv2.cvtColor(o_img, cv2.COLOR_BGR2RGB)
    o_img = o_img.transpose((2, 0, 1))

    input = (np.array([o_img]) / 127.5 - 1.).astype(np.float32)
    outputs = session.run(None, {'input': input})

    concat_img = np.array(outputs[0][0]).copy()
    concat_input = np.array([np.concatenate([concat_img, input[0]], axis=0)]).astype(np.float32)
    outputs = session2.run(None, {'input': concat_input})
    out_img = np.array(outputs[0][0]).transpose((1, 2, 0))
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    out_img = np.around((out_img + 1.) * 127.5).astype(np.uint8)

    cv2.imshow('output', out_img)
    cv2.waitKey(0)