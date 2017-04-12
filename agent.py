import tensorflow as tf
import numpy as np
import collections,os

class agent:
    def __init__(self,actionList,inputSize, nextBlockSize,n_batch, learning_rate, discountRate, saveFreq, saveFolder, memoryLimit, device="/gpu:0"):
        self.actionList = actionList
        self.n_actions = len(actionList)
        self.n_batch = n_batch
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
        self.device=device
        self.thredIndex = -1

        self.init_model()

        return

    def copySettings(self, global_agt, grad_applier, thredIndex):
        self.thredIndex=thredIndex
        self.sync,self.apply_gradients,self.gradients = agt.sync_from(global_agt, grad_applier)
        #self.sess.run(tf.global_variables_initializer())
        return

    def sync_from(self, src_netowrk, grad_applier, name=None):
        src_vars = src_netowrk.get_vars()
        dst_vars = self.get_vars()
        sync_ops = []

        with tf.device(self.device):
            with tf.name_scope(name, "GameACNetwork", []) as name:
                for(src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

            var_refs = [v._ref() for v in self.get_vars()]
            gradients = tf.gradients(self.total_loss, var_refs, gate_gradients=False, aggregation_method=None, colocate_gradients_with_ops=False)
            apply_gradients = grad_applier.apply_gradients( src_netowrk.get_vars(), gradients )
        print(gradients)

        return tf.group(*sync_ops, name=name), apply_gradients, gradients

    def get_vars(self):
        myVars  = []
        myVars += [self.conv1_w, self.conv1_b]
        myVars += [self.conv2_w, self.conv2_b]
        myVars += [self.fc1_w  , self.fc1_b  ]
        myVars += [self.fc2_w  , self.fc2_b  ]
        myVars += [self.fc_value_w  , self.fc_value_b  ]
        myVars += [self.fc_policy_w , self.fc_policy_b ]

        return myVars

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

        scope_name = "net_" + str(self.thredIndex)
        with tf.device(self.device), tf.variable_scope(scope_name) as scope:

            #############################
            ### input variable definition
            self.x  = tf.placeholder(tf.float32, [None, self.inputSize[0], self.inputSize[1]])
            self.a  = tf.placeholder("float", [None, self.n_actions]) # taken action (input for policy)
            self.td = tf.placeholder("float", [None,1]) # temporary difference (R-V) (input for policy)
            self.r  = tf.placeholder("float", [None])
            self.len = tf.placeholder("float", [])
            self.drop= tf.placeholder("float", [])
            self.rewardAvg = tf.placeholder("float", [])
            self.rewardDropAvg = tf.placeholder("float", [])

            #############################
            ### network definition
            h = self.x

            # conv1
            h = tf.expand_dims(h,axis=3)
            sb, sh, sw, sf = h.get_shape()
            self.conv1_w, self.conv1_b = self._conv_variable([3,3,1,32])
            h = self._conv2d(h, self.conv1_w, stride=1) + self.conv1_b
            h = self.leakyReLU(h)

            # conv2
            sb, sh, sw, sf = h.get_shape()
            self.conv2_w, self.conv2_b = self._conv_variable([3,3,sf,128])
            h = self._conv2d(h, self.conv2_w, stride=1) + self.conv2_b
            h = self.leakyReLU(h)

            # fc1
            sb, sh, sw, sf = h.get_shape()
            h = tf.reshape(h,[-1,int(sh)*int(sw)*int(sf)])
            self.fc1_w, self.fc1_b = self._fc_variable([int(sh)*int(sw)*int(sf), 256])
            h = tf.matmul(h, self.fc1_w) + self.fc1_b
            h = self.leakyReLU(h)

            # fc2
            sb, sf = h.get_shape()
            self.fc2_w, self.fc2_b = self._fc_variable([int(sf), 256])
            h = tf.matmul(h, self.fc2_w) + self.fc2_b
            h = self.leakyReLU(h)

            # fc_value
            hv = h
            sb, sf = hv.get_shape()
            self.fc_value_w, self.fc_value_b = self._fc_variable([sf, 1])
            hv = tf.matmul(hv, self.fc_value_w) + self.fc_value_b

            # fc_policy
            hp = h
            sb, sf = hp.get_shape()
            self.fc_policy_w, self.fc_policy_b = self._fc_variable([sf, self.n_actions])
            hp = tf.matmul(hp, self.fc_policy_w) + self.fc_policy_b

            # define v and pi
            sb, sf = hv.get_shape()
            self.v  = tf.reshape(hv,[-1]) # flatten
            self.pi = tf.nn.softmax(hp)

            #############################
            ### loss definition
            self.log_pi      =  tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))
            self.entropy     = -tf.reduce_sum( self.pi * self.log_pi, reduction_indices=1)
            self.policy_loss = -tf.reduce_mean( tf.reduce_sum( tf.multiply( self.log_pi, self.a ), reduction_indices=1 ) * self.td + self.entropy * self.entropy_beta )
            self.value_loss  = 0.5 * tf.reduce_mean(tf.multiply(self.r-self.v,self.r-self.v)/2.)

            self.total_loss = self.policy_loss + self.value_loss

            #############################
            ### optimizer
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.optimizer = optimizer.minimize(self.total_loss)

        #############################
        ### summary
        """
        tf.summary.scalar("loss_total" ,self.total_loss)
        tf.summary.scalar("loss_policy",self.policy_loss)
        tf.summary.scalar("loss_value" ,self.value_loss)
        tf.summary.scalar("Entropy"    ,tf.reduce_sum(self.entropy))
        tf.summary.scalar("TD error"   ,tf.nn.l2_loss(self.td))
        tf.summary.histogram("v"    ,self.v)
        tf.summary.histogram("pi"   ,self.pi)
        tf.summary.scalar("length"     ,self.len)
        tf.summary.scalar("drops"      ,self.drop)
        tf.summary.scalar("rewardAvg"      ,self.rewardAvg)
        tf.summary.scalar("rewardDropAvg"  ,self.rewardDropAvg)
        """
        tf.summary.histogram("conv1_W"   ,self.conv1_w)
        tf.summary.histogram("conv1_b"   ,self.conv1_b)
        tf.summary.histogram("conv2_W"   ,self.conv2_w)
        tf.summary.histogram("conv2_b"   ,self.conv2_b)
        tf.summary.histogram("fc1_W"     ,self.fc1_w)
        tf.summary.histogram("fc1_b"     ,self.fc1_b)
        tf.summary.histogram("fc2_W"     ,self.fc2_w)
        tf.summary.histogram("fc2_b"     ,self.fc2_b)
        tf.summary.histogram("fc_value_W"     ,self.fc_value_w)
        tf.summary.histogram("fc_value_b"     ,self.fc_value_b)
        tf.summary.histogram("fc_policy_W"     ,self.fc_policy_w)
        tf.summary.histogram("fc_policy_b"     ,self.fc_policy_b)

        #############################
        ### saver
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()
        if self.saveFolder: self.writer = tf.summary.FileWriter(self.saveFolder)
        return

    def InitSession(self,init=False):
        #############################
        ### session
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=self.memoryLimit))
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        return

    def selectNextAction(self, state):
        action_prob, value = self.sess.run([self.pi,self.v], feed_dict = {self.x: [state]})
        return np.random.choice(self.actionList, p=action_prob[0]),value

    def selectMaxNextAction(self, state):
        action_prob, value = self.sess.run([self.pi,self.v], feed_dict = {self.x: [state]})
        return self.actionList[np.argmax(action_prob[0])],value

    def clearExperience(self):
        self.experience.clear()
        return

    def storeExperience(self, state_t, action, value, state_tp1, reward, terminal):
        self.experience.append((state_t, self.actionList.index(action), value, state_tp1, reward, terminal))
        return

    def trainFromExperience(self,addSummary=None):
        ######################
        ## Calculation
        batch_x  = []
        batch_a  = []
        batch_td = []
        batch_r  = []
        R = 0.
        for state_t, action, value, state_t1, reward, terminal in reversed(self.experience):
            R = reward + self.discountRate * R
            td = R - value
            a = np.zeros([self.n_actions])
            a[action] = 1
            batch_x.append(state_t)
            batch_a.append(a)
            batch_td.append(td)
            batch_r.append(R)
        ######################
        ## Train
        #_, summary = self.sess.run([self.apply_gradients,self.summary],
        #                            feed_dict={self.x:batch_x, self.a:batch_a, self.td:batch_td, self.r:batch_r, self.len:addSummary["length"],
        #                                       self.drop:addSummary["rewardDrop"],self.rewardDropAvg:addSummary["rewardDropAvg"],self.rewardAvg:addSummary["rewardAvg"]})
        _ = self.sess.run(self.apply_gradients,
                                    feed_dict={self.x:batch_x, self.a:batch_a, self.td:batch_td, self.r:batch_r, self.len:addSummary["length"],
                                               self.drop:addSummary["rewardDrop"],self.rewardDropAvg:addSummary["rewardDropAvg"],self.rewardAvg:addSummary["rewardAvg"]})

        return summary

    def load(self, model_path=None):
        self.saver.restore(self.sess, model_path)
