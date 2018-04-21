from PIL import Image
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import random
from tensorflow.python.framework import graph_util


IMAGE_HEIGHT = 40
IMAGE_WIDTH = 150
MAX_CAPTCHA = 4
CHAR_SET_LEN = 36
IMAGE_NUM = 5156
IMAGE_PATH = 'D:/pimageall/'

fileindex = 0

def get_name_and_image():
    global fileindex
    all_image = os.listdir(IMAGE_PATH)
    random_file = random.randint(0, IMAGE_NUM)
    base = os.path.basename(IMAGE_PATH + all_image[random_file])
    name = os.path.splitext(base)[0].split('_')[2].lower()
    image = Image.open(IMAGE_PATH + all_image[random_file])
    image = np.array(image)
    fileindex += 1
    return name, image


def name2vec(name):
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    for i, c in enumerate(name):
        idx = 0
        if (c >= '0' and c <= '9'):
            idx = i * CHAR_SET_LEN + int(c)
        else:
            idx = i * CHAR_SET_LEN + ord(c) - 97 + 10
        vector[idx] = 1
    return vector


def vec2name(vec):
    name = []
    for i in vec:
        if int(i) < 10:
            name.append(chr(int(i) + 48))
        else:
            name.append(chr(int(i) - 10 + 97))
        pass
    pass
    return "".join(name)


# 生成一个训练batch
def get_next_batch(batch_size=64):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])

    for i in range(batch_size):
        name, image = get_name_and_image()
        batch_x[i, :] = 1*(image.flatten())
        batch_y[i, :] = name2vec(name)
    return batch_x, batch_y

####################################################

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH], name='input_x')
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN], name='input_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name='reshaped_input_x')
    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([5, 5, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([5, 5, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([5, 5, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([5 * 19 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out, name='logits')
    return out


# 训练
def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
            print(step, loss_)

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                # 如果准确率大于50%,保存模型,完成训练
                if acc > 0.99:
                    saver.save(sess, "./crack_capcha.model", global_step=step)
                    break

            step += 1

def crack_captcha():
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        n = 1
        error = 0
        while n <= 2:
            text, image = get_name_and_image()

            image = 1 * (image.flatten())
            print(image)
            predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
            text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})
            vec = text_list[0].tolist()
            predict_text = vec2name(vec)
            if text != predict_text:
                error += 1
                print("ERROR: 正确: {}  预测: {}".format(text, predict_text))
            print("正确: {}  预测: {}".format(text, predict_text))
            n += 1
        print('错误率：' + str(float(error) / IMAGE_NUM))


#crack_captcha()
MODEL_DIR = "model/pb"
MODEL_NAME = "crack_51fp_captcha.pb"
CKPT_MODEL_POINT = './model/51fp/'
def storge_pb():
    if not tf.gfile.Exists(MODEL_DIR):  # 创建目录
        tf.gfile.MakeDirs(MODEL_DIR)
    checkpoint = tf.train.get_checkpoint_state(CKPT_MODEL_POINT)  # 检查目录下ckpt文件状态是否可用
    input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
    output_graph = os.path.join(MODEL_DIR, MODEL_NAME) #PB模型保存路径
    output_node_names = 'logits'  # 原模型输出操作节点的名字
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)  # 得到图、clear_devices ：Whether or not to clear the device field for an `Operation` or `Tensor` during import.

    graph = tf.get_default_graph() #获得默认的图
    input_graph_def = graph.as_graph_def()  #返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        #sess.run("predictions:0", feed_dict={"input_holder:0": [10.0]}) # 测试读出来的模型是否正确，注意这里传入的是输出 和输入 节点的 tensor的名字，不是操作节点的名字

        output_graph_def = graph_util.convert_variables_to_constants(  #模型持久化，将变量值固定
            sess,
            input_graph_def,
            output_node_names.split(",") #如果有多个输出节点，以逗号隔开
        )
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

        for op in graph.get_operations():
            print(op.name, op.values())

if __name__ == '__main__':
    #train_crack_captcha_cnn()
    #crack_captcha()
    storge_pb()

