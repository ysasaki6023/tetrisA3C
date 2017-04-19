import tensorflow as tf
import numpy as np
import collections,os

class agent:
    def __init__(self, actionList, inputSize, nextBlockSize, nBatch, timeStep, learning_rate, discountRate, saveFreq, softTemp, saveFolder, memoryLimit):
        self.actionList = actionList
        self.n_actions = len(actionList)
        self.nBatch = nBatch
        self.inputSize = inputSize # must be tuple (height, width)
        self.nextBlockSize = nextBlockSize # must be tuple (height, width)
        self.experience  = collections.deque(maxlen=1000)
        self.learning_rate = learning_rate
        self.discountRate = discountRate
        self.totalCount = 0
        self.saveFreq = saveFreq
        self.saveFolder = saveFolder
        self.saveModel  = "model.ckpt"
        self.memoryLimit = memoryLimit
        self.entropy_beta = 0.01
        self.timeStep = timeStep
        self.softTemp = softTemp

        self.init_model()
        return

    def leakyReLU(self,x,alpha=0.1):
        return tf.maximum(x*alpha,x)

    def _fc_variable(self, weight_shape):
        input_channels  = int(weight_shape[0])
        output_channels = int(weight_shape[1])
        weight_shape = (input_channels, output_channels)
        d = 1.0 / np.sqrt(input_channels)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
        return weight, bias

    def _conv_variable(self, weight_shape):
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        input_channels  = int(weight_shape[2])
        output_channels = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        d = 1.0 / np.sqrt(input_channels * w * h)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
        return weight, bias

    def _conv2d(self, x, W, stride):
        #print(x.get_shape(),W.get_shape())
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    def init_model(self):

        #############################
        ### network definition
        with tf.variable_scope("net") as scope:

            #############################
            ### input variable definition
            self.x  = tf.placeholder(tf.float32, [self.nBatch, self.timeStep, self.inputSize[0], self.inputSize[1]],name="x")
            self.a  = tf.placeholder(tf.float32, [self.nBatch, self.timeStep, self.n_actions],name="a") # taken action (input for policy)
            self.td = tf.placeholder(tf.float32, [self.nBatch, self.timeStep],name="td") # temporary difference (R-V) (input for policy)
            self.r  = tf.placeholder(tf.float32, [self.nBatch, self.timeStep],name="r")
            self.len = tf.placeholder(tf.float32, [],name="len")
            self.drop= tf.placeholder(tf.float32, [],name="drop")
            self.rewardAvg = tf.placeholder(tf.float32, [],name="rewardAvg")
            self.rewardDropAvg = tf.placeholder(tf.float32, [],name="rewardDropAvg")

            # start
            h = self.x

            h = tf.reshape(h,(self.nBatch*self.timeStep,self.inputSize[0],self.inputSize[1]))

            # conv1
            h = tf.expand_dims(h,axis=3)
            self.conv1_w, self.conv1_b = self._conv_variable([3,3,1,32])
            h = self._conv2d(h, self.conv1_w, stride=1) + self.conv1_b
            h = self.leakyReLU(h)

            # conv2
            self.conv2_w, self.conv2_b = self._conv_variable([3,3,32,64])
            h = self._conv2d(h, self.conv2_w, stride=1) + self.conv2_b
            h = self.leakyReLU(h)

            # conv3
            self.conv3_w, self.conv3_b = self._conv_variable([3,3,64,128])
            h = self._conv2d(h, self.conv3_w, stride=1) + self.conv3_b
            h = self.leakyReLU(h)

            # fc1
            _,sh,sw,sf = [int(a) for a in h.get_shape()]
            h = tf.reshape(h,[self.nBatch * self.timeStep, sh*sw*sf])
            self.fc1_w, self.fc1_b = self._fc_variable([sh*sw*sf, 256])
            h = tf.matmul(h, self.fc1_w) + self.fc1_b
            h = self.leakyReLU(h)

            # LSTM
            h = tf.reshape(h,[self.nBatch , self.timeStep, 256])
            self.lstm = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=False)
            self.initial_lstm_state = tf.placeholder(tf.float32, [self.nBatch, 256*2])
            lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm, h, initial_state = self.initial_lstm_state, time_major = False, scope=scope)

            # fc_value
            hv = h
            hv = tf.reshape(hv,[self.nBatch * self.timeStep, 256])
            self.fc_value_w, self.fc_value_b = self._fc_variable([256, 1])
            hv = tf.matmul(hv, self.fc_value_w) + self.fc_value_b
            hv = tf.reshape(hv,[self.nBatch , self.timeStep, 1])

            # fc_policy
            hp = h
            hp = tf.reshape(hp,[self.nBatch * self.timeStep, 256])
            self.fc_policy_w, self.fc_policy_b = self._fc_variable([256, self.n_actions])
            hp = tf.matmul(hp, self.fc_policy_w) + self.fc_policy_b
            hp = tf.reshape(hp,[self.nBatch , self.timeStep, self.n_actions])

            # define v and pi
            self.v      = tf.reshape(hv,[self.nBatch, self.timeStep]) # flatten
            self.pi     = tf.nn.softmax(hp)
            self.policy = hp

            #############################
            ### loss definition
            self.log_pi      =  tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))
            self.entropy     = -tf.reduce_sum ( self.pi * self.log_pi, reduction_indices=2)
            self.policy_loss = -tf.reduce_mean( tf.reduce_sum( tf.multiply( self.log_pi, self.a ), reduction_indices=2 ) * self.td + self.entropy * self.entropy_beta )
            self.value_loss  = 0.5 * tf.reduce_mean(tf.multiply(self.r-self.v,self.r-self.v)/2.)

            self.l2_loss     = 1e-10 * (   tf.nn.l2_loss(self.conv1_w) + tf.nn.l2_loss(self.conv2_w)    + tf.nn.l2_loss(self.conv3_w)
                                         + tf.nn.l2_loss(self.fc1_w)   + tf.nn.l2_loss(self.fc_value_w) + tf.nn.l2_loss(self.fc_policy_w) )

            self.total_loss = self.policy_loss + self.value_loss + self.l2_loss

            #############################
            ### optimizer
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer = optimizer.minimize(self.total_loss)

            #############################
            ### summary
            tf.summary.scalar("loss_total" ,self.total_loss)
            tf.summary.scalar("loss_policy",self.policy_loss)
            tf.summary.scalar("loss_value" ,self.value_loss)
            tf.summary.scalar("loss_l2"    ,self.l2_loss)
            tf.summary.scalar("Entropy"    ,tf.reduce_sum(self.entropy))
            tf.summary.scalar("TD error"   ,tf.nn.l2_loss(self.td))
            tf.summary.scalar("length"     ,self.len)
            tf.summary.scalar("drops"      ,self.drop)
            tf.summary.scalar("rewardAvg"      ,self.rewardAvg)
            tf.summary.scalar("rewardDropAvg"  ,self.rewardDropAvg)
            tf.summary.histogram("v"    ,self.v)
            tf.summary.histogram("pi"   ,self.pi)
            tf.summary.histogram("conv1_W"   ,self.conv1_w)
            tf.summary.histogram("conv1_b"   ,self.conv1_b)
            tf.summary.histogram("conv2_W"   ,self.conv2_w)
            tf.summary.histogram("conv2_b"   ,self.conv2_b)
            tf.summary.histogram("conv3_W"   ,self.conv3_w)
            tf.summary.histogram("conv3_b"   ,self.conv3_b)
            tf.summary.histogram("fc1_W"     ,self.fc1_w)
            tf.summary.histogram("fc1_b"     ,self.fc1_b)
            tf.summary.histogram("fc_value_W"     ,self.fc_value_w)
            tf.summary.histogram("fc_value_b"     ,self.fc_value_b)
            tf.summary.histogram("fc_policy_W"     ,self.fc_policy_w)
            tf.summary.histogram("fc_policy_b"     ,self.fc_policy_b)

        #############################
        ### saver
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()
        if self.saveFolder: self.writer = tf.summary.FileWriter(self.saveFolder)

        #############################
        ### session
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=self.memoryLimit))
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        return

    def SoftMaxWithTemp(self, x, T=1.):
        x -= np.max(x)
        x  = np.exp(x / T)
        return x / np.sum(x)

    def selectNextAction(self, x, T=1.):
        # x  : (s, h, w)
        xx = np.zeros((self.nBatch, self.timeStep, self.inputSize[0], self.inputSize[1]),dtype=np.float32)
        # xx : (b, s, h, w)
        for i, xItem in enumerate(x):
            xx[0,i,:,:] = xItem
        policy, value = self.sess.run([self.policy,self.v], feed_dict = {self.x: xx, self.initial_lstm_state: np.zeros((self.nBatch, 256*2))})
        policy, value = policy[0,len(x)-1], value[0,len(x)-1]
        action_prob = self.SoftMaxWithTemp(policy,T)
        return np.random.choice(self.actionList, p=action_prob),value

    def selectNextMaxAction(self, x, prev_lstm_state=None):
        # xx : (b, s, h, w)
        xx = np.zeros((self.nBatch, self.timeStep, self.inputSize[0], self.inputSize[1]),dtype=np.float32)
        xx[0,0,:,:] = x[0]
        if type(prev_lstm_state)==type(None) : initState = np.zeros((self.nBatch, 256*2))
        else                                 : initState = prev_lstm_state
        policy, value, state = self.sess.run([self.policy,self.v,self.lstm_state], feed_dict = {self.x: xx, self.initial_lstm_state: initState})
        policy, value = policy[0,len(x)-1], value[0,len(x)-1]
        action_prob = self.SoftMaxWithTemp(policy,T=1.)
        return self.actionList[np.argmax(action_prob)],value,state

    def clearExperience(self):
        self.experience.clear()
        return

    def storeExperience(self, state_t, action, value, state_tp1, reward, terminal):
        self.experience.append((state_t, self.actionList.index(action), value, state_tp1, reward, terminal))
        return

    def calculateValue(self,x):
        # x  : (b, h, w)
        xx = np.array(x)
        xx = np.expand_dims(x,axis=1)
        xx = np.tile(xx,(1,self.timeStep,1,1))
        # xx : (b, s, h, w)
        vv = self.sess.run(self.v, feed_dict={self.x:xx})
        vv = vv[:,0]
        return vv

    def trainFromExperience(self,addSummary=None):
        ######################
        ## Set Batch Index
        batchIdx = np.random.randint(0,len(self.experience)-self.timeStep,self.nBatch)
        batchIdx[0] = len(self.experience)-self.timeStep - 1 # Must include the last one

        ## Calculate Values -> need to be V at T = t+1
        temp_x  = []
        for i in range(len(batchIdx)):
            idx = batchIdx[i] + self.timeStep + 1 # +1 is really important
            if idx >= len(self.experience) : idx = len(self.experience)-1 # if it goes beyond the range, set the last one
            temp_x.append(self.experience[idx][0])

        batch_v = self.calculateValue(temp_x)

        ## Prepare Batch
        batch_x  = np.zeros((self.nBatch, self.timeStep, self.inputSize[0], self.inputSize[1]), dtype=np.float32)
        batch_a  = np.zeros((self.nBatch, self.timeStep, self.n_actions), dtype=np.float32)
        batch_d  = np.zeros((self.nBatch, self.timeStep), dtype=np.float32)
        batch_r  = np.zeros((self.nBatch, self.timeStep), dtype=np.float32)

        for i in range(len(batchIdx)):
            R = batch_v[i]
            for j in range(self.timeStep):
                idx = batchIdx[i] - j

                state_t, action, value, state_tp1, reward, terminal = self.experience[idx]
                if terminal: R = 0. # boundary condition for R

                R = reward + self.discountRate * R
                td = R - value

                a = np.zeros([self.n_actions])
                a[action] = 1

                invJ = self.timeStep - j - 1
                batch_x [i,invJ,:,:] = state_t
                batch_a [i,invJ,:]   = a
                batch_d [i,invJ]     = td
                batch_r [i,invJ]     = R

        ######################
        ## Train
        _, summary = self.sess.run([self.optimizer,self.summary],
                                    feed_dict={self.x:batch_x, 
                                               self.a:batch_a, 
                                               self.td:batch_d, 
                                               self.r:batch_r, 
                                               self.initial_lstm_state: np.zeros((self.nBatch, 256*2)),
                                               self.len:addSummary["length"],
                                               self.drop:addSummary["rewardDrop"],
                                               self.rewardDropAvg:addSummary["rewardDropAvg"],
                                               self.rewardAvg:addSummary["rewardAvg"]})

        return summary

    def load(self, model_path=None):
        self.saver.restore(self.sess, model_path)
