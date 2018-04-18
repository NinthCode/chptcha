#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Nicot
# @Date:   2018-04-16 18:02:09
# @Last Modified by:   Nicot
# @Last Modified time: 2018-04-16 18:02:10
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import random
import common

sys.path.append('../util/')
from logger import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

log = Logger('CnnTrain')


class CnnTrain:
    def __init__(self, image_num, image_path, image_height, image_width,
                 captcha_char_num, char_set_len, batch_size, storge_threshold,
                 name2vec, vec2name, model_storge_path='./model/',
                 model_storge_prefix='cnntrain.model'):
        self._image_num = image_num
        self._image_path = image_path
        self._image_height = image_height
        self._image_width = image_width
        self._captcha_char_num = captcha_char_num
        self._char_set_len = char_set_len
        self._batch_size = batch_size
        self._storge_threshold = storge_threshold
        self._name2vec = name2vec
        self._vec2name = vec2name
        self._X = tf.placeholder(tf.float32, [None, image_height * image_width], name='input_x')
        self._Y = tf.placeholder(tf.float32, [None, captcha_char_num * char_set_len], name='imput_y')
        self._keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self._fcl_x = image_height / 8 if image_height % 8 != 0 else int(image_height / 8) + 1
        self._fcl_y = image_width / 8 if image_width % 8 != 0 else int(image_width / 8) + 1
        self._model_storge_name = model_storge_path + model_storge_prefix
        if not tf.gfile.Exists(model_storge_path):  # 创建目录
            tf.gfile.MakeDirs(model_storge_path)
        pass

    pass

    def _get_name_and_image(self):
        all_image = os.listdir(self._image_path)
        random_file = random.randint(0, self._image_num)
        base = os.path.basename(self._image_path + all_image[random_file])
        name = os.path.splitext(base)[0]
        image = Image.open(self._image_path + all_image[random_file])
        image = np.array(image)
        return name, image

    pass

    # 生成一个训练batch
    def _get_next_batch(self, batch_size=64):
        batch_x = np.zeros([batch_size, self._image_height * self._image_width])
        batch_y = np.zeros([batch_size, self._captcha_char_num * self._char_set_len])
        for i in range(batch_size):
            name, image = self._get_name_and_image()
            batch_x[i, :] = 1 * (image.flatten())
            batch_y[i, :] = self._name2vec(name)
        pass
        return batch_x, batch_y

    pass

    # 定义CNN
    def _crack_captcha_cnn(self, w_alpha=0.01, b_alpha=0.1):
        x = tf.reshape(self._X, shape=[-1, self._image_height, self._image_width, 1], name='reshaped_input_x')
        # 3 conv layer
        w_c1 = tf.Variable(w_alpha * tf.random_normal([5, 5, 1, 32]))
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self._keep_prob)

        w_c2 = tf.Variable(w_alpha * tf.random_normal([5, 5, 32, 64]))
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self._keep_prob)

        w_c3 = tf.Variable(w_alpha * tf.random_normal([5, 5, 64, 64]))
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self._keep_prob)

        # Fully connected layer
        w_d = tf.Variable(w_alpha * tf.random_normal([self._fcl_x * self._fcl_y * 64, 1024]))
        b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, self._keep_prob)

        w_out = tf.Variable(w_alpha * tf.random_normal([1024, self._captcha_char_num * self._char_set_len]))
        b_out = tf.Variable(b_alpha * tf.random_normal([self._captcha_char_num * self._char_set_len]))
        out = tf.add(tf.matmul(dense, w_out), b_out, name='logits')
        return out

    pass

    # 训练
    def _train_crack_captcha_cnn(self):
        output = self._crack_captcha_cnn()
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=self._Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        predict = tf.reshape(output, [-1, self._captcha_char_num, self._char_set_len])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(self._Y, [-1, self._captcha_char_num, self._char_set_len]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            step = 0
            while True:
                batch_x, batch_y = self._get_next_batch(64)
                _, loss_ = sess.run([optimizer, loss],
                                    feed_dict={self._X: batch_x, self._Y: batch_y, self._keep_prob: 0.5})
                log.info('train step: ' + str(step) + ', loss: ' + loss_)
                # 每100 step计算一次准确率
                if step % 100 == 0:
                    batch_x_test, batch_y_test = self._get_next_batch(100)
                    acc = sess.run(accuracy,
                                   feed_dict={self._X: batch_x_test, self._Y: batch_y_test, self._keep_prob: 1.})
                    print(step, acc)
                    # 如果准确率大于threshold,保存模型,完成训练
                    if acc > self._storge_threshold:
                        saver.save(sess, self._model_storge_name, global_step=step)
                        break
                    pass
                pass
            step += 1
            pass
        pass

    pass

    def train(self):
        self._train_crack_captcha_cnn()


pass


class Prediction:
    def __init__(self, captcha_char_num, char_set_len, image_width, image_height,
                 name2vec, vec2name, model_path='./model/'):
        self._captcha_char_num = captcha_char_num
        self._char_set_len = char_set_len
        self._name2vec = name2vec
        self._vec2name = vec2name
        self._fcl_x = image_height / 8 if image_height % 8 != 0 else int(image_height / 8) + 1
        self._fcl_y = image_width / 8 if image_width % 8 != 0 else int(image_width / 8) + 1
        self._model_path = model_path
        self._init()

    pass

    def _init(self):
        graph = tf.Graph()
        with graph.as_default():
            self._session = tf.Session()
            latest_checkpoint = tf.train.latest_checkpoint(self._model_path)
            if latest_checkpoint:
                head, tail = os.path.split(latest_checkpoint)
                tf.train.import_meta_graph(os.path.join(self._model_path, tail + ".meta"))
                tf.train.Saver().restore(self._session, latest_checkpoint)
                self._X = graph.get_tensor_by_name('input_x:0')
                self._Y = graph.get_tensor_by_name('input_y:0')
                self._keep_prob = graph.get_tensor_by_name("keep_prob:0")
                logits = graph.get_tensor_by_name('logits:0')
                self._predict = tf.argmax(tf.reshape(logits, [-1, self._captcha_char_num, self._char_set_len]), 2)
            pass
        pass

    pass

    def run(self, image):
        image = np.array(image)
        image = 1 * (image.flatten())
        text_list = self._session.run(self._predict, feed_dict={self._X: [image], self._keep_prob: 1})
        vec = text_list[0].tolist()
        return self._vec2name(vec)

    pass


pass
if __name__ == '__main__':
    pt = Prediction(4, 36, 150, 40, common.name2vec, common.vec2name, model_path='./')
    print(pt.run(Image.open('D:/testsamples/1/20180415210248_61_J8XQ_102583705.png')))
    print(pt.run(Image.open('D:/testsamples/0/20180415205722_13_YYM6_102570056.png')))
    print(pt.run(Image.open('D:/testsamples/0/20180415205853_27_RXG6_102573639.png')))
    print(pt.run(Image.open('D:/testsamples/0/20180415210057_45_9STB_102578751.png')))

