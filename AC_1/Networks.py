# Networks
import os
import tensorflow as tf

from tensorflow.keras.layers import Dense


class ActorCriticNetwork(tf.keras.Model):
    def __init__(self, n_actions,fc1_dims= 32,fc2_dims=32,
    name = 'actor_critic',checkpoint_dir = 'tmp/actor_critic/'):
        super(ActorCriticNetwork,self).__init__()

        self.fc1_dims = fc1_dims  #number of dimensiion on first fully connected layer
        self.fc2_dims = fc2_dims # num dims second fully connected layer
        self.n_actions = n_actions # action space
        self.model_name = name #model name
        self.checkpoint_dir = checkpoint_dir #where to save model checkpoints
        self.checkpoint_file = os.path.join(self.checkpoint_dir+name+'_ac') #file name

        self.fc1 = Dense(self.fc1_dims, activation = 'relu',input_shape=(6,),
                         kernel_regularizer = tf.keras.regularizers.L2(0.01))
        self.fc2 = Dense(self.fc2_dims, activation = 'relu',
                         kernel_regularizer = tf.keras.regularizers.L2(0.01))
        self.v = Dense(1,activation=None)
        self.pi = Dense(self.n_actions,activation='softmax')
    
    def call(self,state):
        value = self.fc1(state)
        value = self.fc2(value)

        v = self.v(value)
        pi = self.pi(value)
        
        return v, pi
    



