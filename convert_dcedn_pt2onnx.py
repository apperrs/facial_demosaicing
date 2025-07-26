import os.path

import numpy as np
import torch.nn

abs_file = os.path.abspath(__file__)
abs_dir = abs_file[:abs_file.rfind('\\')] if os.name == 'nt' else abs_file[:abs_file.rfind(r'/')]
model = torch.load(abs_dir + '/checkpoints/save.pt', map_location=torch.device('cpu'))

input_names = ['input']
output_names = ['output']

x = np.zeros((1, 6, 256, 256), dtype=np.float32)
x = torch.from_numpy(x)

torch.onnx.export(model, x, 'dcedn.onnx', input_names=input_names, output_names=output_names, verbose='True')

