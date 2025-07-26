import json
import os.path
import random
import time
import cv2
from options.train_options import TrainOptions
from data import create_dataset
import models.cycle_gan_model as cycle_gan_model
from util.visualizer import Visualizer

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

if __name__ == '__main__':
    modified_path = '../../../FaceMosaicClean/CASIA-WebFace/'
    data_tmp_path = 'datasets_mosaic'

    if not os.path.exists(data_tmp_path):
        os.mkdir(data_tmp_path)
    if not os.path.exists(data_tmp_path + '/trainA'):
        os.mkdir(data_tmp_path + '/trainA')
    for f in os.listdir(data_tmp_path + '/trainA'):
        os.remove(data_tmp_path + '/trainA/' + f)
    if not os.path.exists(data_tmp_path + '/trainB'):
        os.mkdir(data_tmp_path + '/trainB')
    for f in os.listdir(data_tmp_path + '/trainB'):
        os.remove(data_tmp_path + '/trainB/' + f)

    opt = TrainOptions().parse()
    opt.dataroot = data_tmp_path
    opt.direction = "AtoB"
    opt.batch_size = 1

    model = cycle_gan_model.CycleGANModel(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        model.update_learning_rate()

        for dir in os.listdir(modified_path):
            for f in os.listdir(data_tmp_path + '/trainA'):
                os.remove(data_tmp_path + '/trainA/' + f)
            for f in os.listdir(data_tmp_path + '/trainB'):
                os.remove(data_tmp_path + '/trainB/' + f)
            for file in os.listdir(modified_path + dir + '/'):
                if 'json' in file:
                    continue
                img = cv2.imread(modified_path + dir + '/' + file)
                o_img = img.copy()

                if not os.path.exists(modified_path + '/' + dir + '/' + file + '.json'):
                    continue
                dets = json.loads(open(modified_path + '/' + dir + '/' + file + '.json', 'r').read())
                x1, y1, x2, y2 = 0,0,0,0
                for det in dets:
                    x1, y1, x2, y2 = map(int, det['xyxy'])
                    det['xyxy'] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                a_img = draw_mask(img.copy(), x1, y1, x2, y2, block_size=random.randint(6, 19))
                a_img = a_img[y1:y2, x1:x2]

                b_img = img[y1:y2, x1:x2]
                a_img = cv2.resize(a_img, (256, 256))
                b_img = cv2.resize(b_img, (256, 256))
                cv2.imwrite(data_tmp_path + '/trainA/' + file.replace('.jpg', '.png'), a_img)
                cv2.imwrite(data_tmp_path + '/trainB/' + file.replace('.jpg', '.png'), b_img)

                dataset = create_dataset(opt)
                dataset_size = len(dataset)
                print('The number of training images = %d' % dataset_size)
                for i, data in enumerate(dataset):
                    iter_start_time = time.time()
                    if total_iters % opt.print_freq == 0:
                        t_data = iter_start_time - iter_data_time

                    total_iters += opt.batch_size
                    epoch_iter += opt.batch_size
                    model.set_input(data)
                    model.optimize_parameters()

                    if total_iters % opt.display_freq == 0:
                        save_result = total_iters % opt.update_html_freq == 0
                        model.compute_visuals()
                        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                    if total_iters % opt.print_freq == 0:
                        losses = model.get_current_losses()
                        t_comp = (time.time() - iter_start_time) / opt.batch_size
                        visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                        if opt.display_id > 0:
                            visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                    if total_iters % opt.save_latest_freq == 0:
                        print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                        save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                        model.save_networks(save_suffix)

                    iter_data_time = time.time()
                if epoch % opt.save_epoch_freq == 0:
                    print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                    model.save_networks('latest')
                    model.save_networks(epoch)

                print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))