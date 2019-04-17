import tensorflow as tf
from lanenet_model import lanenet_hnet_model
from data_provider import lanenet_hnet_data_processor
import numpy as np
import cv2
import glob

batch_size = 1
tensor_in = tf.placeholder(dtype=tf.float32, shape=[batch_size, 64, 128, 3])
gt_label_pts = tf.placeholder(dtype=tf.float32, shape=[None, 3])

net = lanenet_hnet_model.LaneNetHNet(phase=tf.constant('train', tf.string))
c_loss, coef = net.compute_loss(tensor_in, gt_label_pts=gt_label_pts, name='hnet')

saver = tf.train.Saver()

train_dataset = lanenet_hnet_data_processor.DataSet(glob.glob('./data/tusimple_data/*.json'))

optimizer = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9).minimize(loss=c_loss, var_list=tf.trainable_variables())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(10000):
        image, label_pts = train_dataset.next_batch(batch_size)
        label_pts = label_pts[0]
        print(label_pts)
        image = np.array(image)
        image.at = 0
        loss, coefficient = sess.run([c_loss, coef], feed_dict={tensor_in: image, gt_label_pts: label_pts})
        if epoch % 50 == 0:
            print('epoch[{}], hnet training loss = {}'.format(epoch, loss))
        epoch += 1
        if epoch % 1000 == 0:
            R = np.zeros([3, 3], np.float32)
            R[0, 0] = coefficient[0]
            R[0, 1] = coefficient[1]
            R[0, 2] = coefficient[2]
            R[1, 1] = coefficient[3]
            R[1, 2] = coefficient[4]
            R[2, 1] = coefficient[5]
            R[2, 2] = 1
            warp_image = cv2.warpPerspective(image[0], R, dsize=(image[0].shape[1], image[0].shape[0]))
            cv2.imwrite("src.jpg", image[0])
            cv2.imwrite("ret.jpg", warp_image)
            saver.save(sess=sess, save_path='./model/hnet/hnet', global_step=epoch)