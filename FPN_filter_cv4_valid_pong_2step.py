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
            self.imageIn = tf.reshape(self.inputs, [-1, (retro_step + 1) * 4, 3 * 8, 3])
            

            self.conv1 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.imageIn, num_outputs=128, kernel_size=[2, 2], stride=[1, 1], padding='SAME')

#            print "conv1 shape, ", self.conv1.get_shape() 
            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv1, num_outputs=128, kernel_size=[3, 3], stride=[1, 1], padding='SAME')
#            print "conv2 shape, ", self.conv2.get_shape()

            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv2, num_outputs=128, kernel_size=[3, 3], stride=[1, 1], padding='SAME')

#            self.conv4 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv3, num_outputs=128, kernel_size=[3, 3], stride=[1, 1], padding='SAME')

#            self.conv5 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv4, num_outputs=128, kernel_size=[3, 3], stride=[1, 1], padding='SAME')

#            self.conv6 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv5, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='SAME')

#            self.conv7 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.conv6, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='SAME')




            
#            hidden1 = slim.fully_connected(slim.flatten(self.imageIn), 750, activation_fn=tf.nn.elu, weights_initializer=normalized_columns_initializer(0.1))

            hidden1 = slim.fully_connected(slim.flatten(self.conv3), 750, activation_fn=tf.nn.elu)


            #Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(512,state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden1, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 512])


            hidden2 = rnn_out




#            hidden2 = slim.fully_connected(slim.flatten(hidden1), 750, activation_fn=tf.nn.elu)


            self.predicted_observation = slim.fully_connected(hidden2, 288, activation_fn=None, weights_initializer=normalized_columns_initializer(0.1))
            self.predicted_reward = slim.fully_connected(hidden2, 1, activation_fn=None, weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None)

            self.predicted_done = slim.fully_connected(hidden2, 1, activation_fn=tf.nn.sigmoid, weights_initializer=normalized_columns_initializer(0.1), biases_initializer=None)

 

            
            #self.true_observation = tf.placeholder(shape=[None, 32], dtype=tf.float32)
            self.true_observation = tf.placeholder(dtype=tf.float32)
            self.true_reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.true_done = tf.placeholder(shape=[None, 1], dtype=tf.float32)

            self.observation_loss = tf.reduce_sum(tf.square(self.true_observation - self.predicted_observation))
            self.reward_loss = tf.reduce_sum(tf.square(self.true_reward - self.predicted_reward))
            self.done_loss = tf.reduce_sum(-tf.log(tf.multiply(self.predicted_done, self.true_done) + tf.multiply(1 - self.predicted_done, 1 - self.true_done)))
            self.loss = tf.reduce_mean(self.observation_loss + 1000 * self.reward_loss + self.done_loss)
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.gradients = tf.gradients(self.loss, local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
                
            #Apply local updates to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'worker')
            self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))






def DAD(history_obs, next_obs, reward, done, predicted_obs):
    history_obs.pop(0)
    for i in range(len(history_obs)):
        history_obs[i][-2] = predicted_obs[i]
    next_obs.pop(0)
    done.pop(0)
    predicted_done.pop(0)
    return history_obs, next_obs, reward, done



h_size = 256


config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config = config)

with tf.device("/gpu:0"):

    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    retro_step = 5
    FPN = FramePrediction_Network(h_size, retro_step, trainer, 'worker')
    num_epochs = 50
    data_size = 4000
    train_data_size = 3600
    buff = 1000
    test_data_size = 400
    pred_step = 2
    data_path = 'PongFF/'
    sess.run(tf.global_variables_initializer())



    #read data 
    past_history= []
    episode_next = []
    episode_reward = []
    episode_done = []
    print 'reading data'
    avg = []
    for data_index in range(data_size):
        #print 'reading ' + str(data_index) + ' file'
        data_index += buff

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
            print "discard"
            train_data_size -= 1
            test_data_size -= 1
            continue

        one_past_history = []
        avg_dis = []
        for i in range(retro_step, len(one_episode_buffer)):
            one_past = np.array([np.array(one_episode_buffer[j]).flatten() for j in range(i - retro_step, i)])

            #one_past = np.array(one_episode_buffer[i - 1])
            #print out the difference between each adjacent frame
            res = np.linalg.norm(np.array(one_episode_buffer[i - 1]) - np.array(one_episode_buffer[i - 2]))

            avg_dis.append(res * res)

            one_action = one_episode_action[i - 1 : i]
            one_action = np.full(one_past[0].shape, one_action)
            #one_action = np.full(one_past.shape, one_action)
            one_moment = np.vstack((one_past, one_action))

            one_past_history.append(one_moment)
            #one_past_history.append(one_past)
 
        next_observations = np.array([np.array(one_episode_buffer[j]).flatten() for j in range(retro_step, len(one_episode_buffer))])

       #cheat training
#        next_observations = np.array([np.array(one_episode_buffer[j]).flatten() for j in range(retro_step - 1, len(one_episode_buffer) - 1)])



        rewards = np.vstack(one_episode_reward[retro_step - 1 : -1])
        dones = np.vstack(one_episode_done[retro_step - 1 : -1])
        one_past_history = np.array(one_past_history)
        
        #print 'data_index: ' + str(data_index), np.mean(avg_dis)
        avg.append(np.mean(avg_dis))



        past_history.append(one_past_history)
        episode_next.append(next_observations)
        episode_reward.append(rewards)
        episode_done.append(dones)

    print np.mean(avg)

    
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

        batch_rnn_state = FPN.state_init

        for data_index in range(train_data_size):            
            #print past_history.shape, next_observations.shape, rewards.shape, dones.shape
            #print 'training ' + str(data_index) + ' file'
            sudo_data_index = data_index

            history_container = past_history[sudo_data_index]
            next_container = episode_next[sudo_data_index]
            reward_container = episode_reward[sudo_data_index]
            done_container = episode_done[sudo_data_index]


            acc_loss = 0
            acc_o_l = 0
            acc_r_l = 0
            acc_d_l = 0

            for j in range(pred_step):

                feed_dict = {FPN.inputs:hisotry_container, FPN.true_observation:next_container, FPN.true_reward:reward_container, FPN.true_done:done_container, FPN.state_in[0]:batch_rnn_state[0], FPN.state_in[1]:batch_rnn_state[1]}

                po, o_l, r_l, d_l, loss, batch_rnn_state, _ = sess.run([FPN.predicteed_observation, FPN.observation_loss, FPN.reward_loss, FPN.done_loss, FPN.loss, FPN.state_out, FPN.apply_grads], feed_dict=feed_dict)

                history_container, next_container, reward_container, done_container = DAD(history_container, next_observations, reward_container, done_container, po)
            

                len_episode_buffer = len(past_history[sudo_data_index])

                acc_loss += loss / len_episode_buffer
                acc_o_l += o_l / len_episode_buffer
                acc_r_l += r_l / len_episode_buffer
                acc_d_l += d_l / len_episode_buffer

                

            epoch_loss.append(acc_loss / pred_step)
            epoch_ob_loss.append(acc_o_l / pred_step)
            epoch_reward_loss.append(acc_r_l / pred_step)
            epoch_done_loss.append(acc_d_l / pred_step)


        if epoch % 5 == 0 and epoch != 0:
            summary = tf.Summary()
            l = np.mean(epoch_loss)
            o_l = np.mean(epoch_ob_loss)
            r_l = np.mean(epoch_reward_loss)
            d_l = np.mean(epoch_done_loss)
            print "training on the " + str(epoch) + " epoch and the error is :" 
            print l, o_l, r_l, d_l
            #summary.value.add(tag='Losses/Prediction Loss', simple_value=float(l))
            #summary.value.add(tag='Losses/Observation Loss', simple_value=float(o_l))
            #summary.value.add(tag='Losses/Reward Loss', simple_value=float(r_l))
            #summary.value.add(tag='Losses/Done Loss', simple_value=float(d_l))
            #summary_writer = tf.summary.FileWriter("train_FPN")
            #summary_writer.add_summary(summary, epoch)
            #summary_writer.flush()

            #print out the validation error

            val_epoch_loss = []
            val_epoch_ob_loss = []
            val_epoch_reward_loss = []
            val_epoch_done_loss = []

            batch_rnn_state = FPN.state_init

            for data_index in range(test_data_size):            
                #print past_history.shape, next_observations.shape, rewards.shape, dones.shape
                #print 'training ' + str(data_index) + ' file'
                sudo_data_index = data_index + train_data_size
                acc_loss = 0
                acc_o_l = 0
                acc_r_l = 0
                acc_d_l = 0
                for j in range(pred_step):
                    feed_dict = {FPN.inputs:past_history[sudo_data_index], FPN.true_observation:episode_next[sudo_data_index], FPN.true_reward:episode_reward[sudo_data_index], FPN.true_done:episode_done[sudo_data_index], FPN.state_in[0]:batch_rnn_state[0], FPN.state_in[1]:batch_rnn_state[1]}

                    o_l, r_l, d_l, loss, batch_rnn_state = sess.run([FPN.observation_loss, FPN.reward_loss, FPN.done_loss, FPN.loss, FPN.state_out], feed_dict=feed_dict)

                    len_episode_buffer = len(past_history[sudo_data_index])

                    acc_loss += loss / len_episode_buffer
                    acc_o_l += o_l / len_episode_buffer
                    acc_r_l += r_l / len_episode_buffer
                    acc_d_l += d_l / len_episode_buffer

                val_epoch_loss.append(acc_loss / pred_step)
                val_epoch_ob_loss.append(acc_o_l / pred_step)
                val_epoch_reward_loss.append(acc_r_l / pred_step)
                val_epoch_done_loss.append(acc_d_l / pred_step)

            l = np.mean(val_epoch_loss)
            o_l = np.mean(val_epoch_ob_loss)
            r_l = np.mean(val_epoch_reward_loss)
            d_l = np.mean(val_epoch_done_loss)

            print "Validate error is: " 
            print l, o_l, r_l, d_l

















                
            
            
