import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import os


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class FramePrediction_Network(object):
    def __init__(self, h_size, retro_step, trainer, scope):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(tf.float32)
            #self.imageIn = tf.reshape(self.inputs, [-1, 9 * (retro_step + 1), 9, 32])
            #self.imageIn = tf.reshape(self.inputs, [-1, 9, 9, 32])
            self.imageIn = tf.reshape(self.inputs, [-1, (retro_step + 1) * 16, 16, 1])
            

            self.conv1 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.imageIn, num_outputs=32, kernel_size=[3, 3], stride=[2, 2], padding='SAME')
            print "conv1 shape, ", self.conv1.get_shape()
            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv1, num_outputs=16, kernel_size=[3, 3], stride=[1, 1], padding='SAME')
            print "conv2 shape, ", self.conv2.get_shape()

#            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv2, num_outputs=64, kernel_size=[2, 2], stride=[1, 1], padding='SAME')
#            self.conv4 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv3, num_outputs=32, kernel_size=[3, 3], stride=[1, 1], padding='SAME')
#            self.conv5 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv4, num_outputs=32, kernel_size=[3, 3], stride=[1, 1], padding='SAME')
#            self.conv6 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv5, num_outputs=16, kernel_size=[3, 3], stride=[1, 1], padding='SAME')
#            self.conv7 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv6, num_outputs=16, kernel_size=[3, 3], stride=[1, 1], padding='SAME')




            
#            hidden1 = slim.fully_connected(slim.flatten(self.conv2), h_size, activation_fn=tf.nn.elu)

            hidden1 = slim.fully_connected(slim.flatten(self.conv2), h_size * 8, activation_fn=tf.nn.elu)

            
            hidden2 = slim.fully_connected(hidden1, h_size * 4, activation_fn=tf.nn.elu)

            hidden3 = slim.fully_connected(hidden2, h_size * 2, activation_fn=tf.nn.elu)
#            hidden4 = slim.fully_connected(hidden3, h_size / 8, activation_fn=tf.nn.elu)

            self.predicted_observation = slim.fully_connected(hidden3, 256, activation_fn=None, weights_initializer=normalized_columns_initializer(0.1))
            self.predicted_reward = slim.fully_connected(hidden3, 1, activation_fn=None, weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None)

            self.predicted_done = slim.fully_connected(hidden3, 1, activation_fn=tf.nn.sigmoid, weights_initializer=normalized_columns_initializer(0.1), biases_initializer=None)

 

            
            #self.true_observation = tf.placeholder(shape=[None, 32], dtype=tf.float32)
            self.true_observation = tf.placeholder(dtype=tf.float32)
            self.true_reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.true_done = tf.placeholder(shape=[None, 1], dtype=tf.float32)

            self.observation_loss = tf.reduce_sum(tf.square(self.true_observation - self.predicted_observation))
            self.reward_loss = tf.reduce_sum(tf.square(self.true_reward - self.predicted_reward))
            self.done_loss = tf.reduce_sum(-tf.log(tf.multiply(self.predicted_done, self.true_done) + tf.multiply(1 - self.predicted_done, 1 - self.true_done)))
            self.loss = tf.reduce_mean(self.observation_loss + self.reward_loss + self.done_loss)
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.gradients = tf.gradients(self.loss, local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
                
            #Apply local updates to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'worker')
            self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


h_size = 256


config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config = config)

with tf.device("/gpu:0"):

    trainer = tf.train.AdamOptimizer(learning_rate=5e-5)
    retro_step = 3
    FPN = FramePrediction_Network(h_size, retro_step, trainer, 'worker')
    num_epochs = 100
    data_size = 1039
    train_data_size = 800
    test_data_size = data_size - 800
    data_path = '../frames/'
    sess.run(tf.global_variables_initializer())



    #read data 
    past_history= []
    episode_next = []
    episode_reward = []
    episode_done = []
    print 'reading data'
    for data_index in range(data_size):
        #print 'reading ' + str(data_index) + ' file'
        data_path_for_one_ep = data_path + 'episode_' + str(data_index) + '/'
        one_episode_buffer = []
        one_episode_action = []
        one_episode_reward = []
        one_episode_done = []

        action_file = data_path_for_one_ep + 'action.txt'
        reward_file = data_path_for_one_ep + 'reward.txt'
        done_file = data_path_for_one_ep + 'done.txt'

        f = open(action_file, 'r')
        text = f.readlines()
        for line in text:
            one_episode_action.append(float(line.strip()))
            
        f = open(reward_file, 'r')
        text = f.readlines()
        for line in text:
            one_episode_reward.append(float(line.strip()))

        f = open(done_file, 'r')
        text = f.readlines()
        for line in text:
            one_episode_done.append(bool(line.strip()))
        
        episode_len = len(one_episode_action)

        
        for image_index in range(episode_len):
            file_path = data_path_for_one_ep + 'image_' + str(image_index) + '.npy'
            im = np.load(file_path)
            one_episode_buffer.append(im)
        if len(one_episode_buffer) <= retro_step:
            continue

        one_past_history = []
        for i in range(retro_step, len(one_episode_buffer)):
            one_past = np.array([np.array(one_episode_buffer[j]).flatten() for j in range(i - retro_step, i)])
            #one_past = np.array(one_episode_buffer[i - 1])
            #print out the difference between each adjacent frame
            #print np.linalg.norm(np.array(one_episode_buffer[i - 1]) - np.array(one_episode_buffer[i - 2]))

            one_action = one_episode_action[i - 1 : i]
            one_action = np.full(one_past[0].shape, one_action)
            #one_action = np.full(one_past.shape, one_action)
            one_moment = np.vstack((one_past, one_action))

            one_past_history.append(one_moment)
            #one_past_history.append(one_past)
        next_observations = np.array([np.array(one_episode_buffer[j]).flatten() for j in range(retro_step, len(one_episode_buffer))])

        #next_observations = np.array([np.array(one_episode_buffer[j]).flatten() for j in range(retro_step - 1, len(one_episode_buffer) - 1)])


        rewards = np.vstack(one_episode_reward[retro_step - 1 : -1])
        dones = np.vstack(one_episode_done[retro_step - 1 : -1])
        one_past_history = np.array(one_past_history)


        past_history.append(one_past_history)
        episode_next.append(next_observations)
        episode_reward.append(rewards)
        episode_done.append(dones)




    #shuffle data

    indices = [i for i in range(len(past_history))]
    np.random.shuffle(indices)
    past_history = [past_history[i] for i in indices]
    episode_next = [episode_next[i] for i in indices]
    episode_reward = [episode_reward[i] for i in indices]
    episode_done = [episode_done[i] for i in indices]



    #Training
    print "training..."
               
    for epoch in range(num_epochs):
        epoch_loss = []
        epoch_ob_loss = []
        epoch_reward_loss = []
        epoch_done_loss = []

        for data_index in range(train_data_size):            
            #print past_history.shape, next_observations.shape, rewards.shape, dones.shape
            #print 'training ' + str(data_index) + ' file'
            sudo_data_index = data_index
            feed_dict = {FPN.inputs:past_history[sudo_data_index], FPN.true_observation:episode_next[sudo_data_index], FPN.true_reward:episode_reward[sudo_data_index], FPN.true_done:episode_done[sudo_data_index]}

            o_l, r_l, d_l, loss, _ = sess.run([FPN.observation_loss, FPN.reward_loss, FPN.done_loss, FPN.loss, FPN.apply_grads], feed_dict=feed_dict)

            len_episode_buffer = len(past_history[sudo_data_index])

            loss /= (len_episode_buffer - retro_step)
            o_l /= (len_episode_buffer - retro_step)
            r_l /= (len_episode_buffer - retro_step)
            d_l /= (len_episode_buffer - retro_step)

            epoch_loss.append(loss)
            epoch_ob_loss.append(o_l)
            epoch_reward_loss.append(r_l)
            epoch_done_loss.append(d_l)
        if epoch % 5 == 0 and epoch != 0:
            summary = tf.Summary()
            l = np.mean(epoch_loss)
            o_l = np.mean(epoch_ob_loss)
            r_l = np.mean(epoch_reward_loss)
            d_l = np.mean(epoch_done_loss)
            print "training on the " + str(epoch) + " epoch and the error is :" 
            print l, o_l, r_l, d_l
            summary.value.add(tag='Losses/Prediction Loss', simple_value=float(l))
            summary.value.add(tag='Losses/Observation Loss', simple_value=float(o_l))
            summary.value.add(tag='Losses/Reward Loss', simple_value=float(r_l))
            summary.value.add(tag='Losses/Done Loss', simple_value=float(d_l))
            summary_writer = tf.summary.FileWriter("train_FPN")
            summary_writer.add_summary(summary, epoch)
            summary_writer.flush()


    #validate
    print "Testing"

    epoch_loss = []
    epoch_ob_loss = []
    epoch_reward_loss = []
    epoch_done_loss = []

    for data_index in range(test_data_size):            
        #print past_history.shape, next_observations.shape, rewards.shape, dones.shape
        #print 'training ' + str(data_index) + ' file'
        sudo_data_index = data_index + train_data_size
        feed_dict = {FPN.inputs:past_history[sudo_data_index], FPN.true_observation:episode_next[sudo_data_index], FPN.true_reward:episode_reward[sudo_data_index], FPN.true_done:episode_done[sudo_data_index]}

        o_l, r_l, d_l, loss = sess.run([FPN.observation_loss, FPN.reward_loss, FPN.done_loss, FPN.loss], feed_dict=feed_dict)

        len_episode_buffer = len(past_history[sudo_data_index])

        loss /= (len_episode_buffer - retro_step)
        o_l /= (len_episode_buffer - retro_step)
        r_l /= (len_episode_buffer - retro_step)
        d_l /= (len_episode_buffer - retro_step)

        #print "test on the " + str(data_index) + " file and the error is :" 
        #print loss, o_l, r_l, d_l
        epoch_loss.append(loss)
        epoch_ob_loss.append(o_l)
        epoch_reward_loss.append(r_l)
        epoch_done_loss.append(d_l)

    l = np.mean(epoch_loss)
    o_l = np.mean(epoch_ob_loss)
    r_l = np.mean(epoch_reward_loss)
    d_l = np.mean(epoch_done_loss)

    print "final error is: " 
print l, o_l, r_l, d_l













                
            
            
