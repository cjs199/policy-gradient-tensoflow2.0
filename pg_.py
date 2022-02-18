from tensorflow.keras import layers, optimizers, Model
import tensorflow as tf

# 1. 构建模型
class PG(Model):

    # 构造器
    def __init__(self, act_dim, lr):
        super(PG, self).__init__()
        self.lr = lr
        self.act_dim = act_dim
        self.ds1 = layers.Dense(256, activation='relu')
        self.ds2 = layers.Dense(64, activation='relu')
        self.ds3 = layers.Dense(act_dim, activation=tf.nn.softmax)

    # 前向传播  : 通过输入的游戏画面,进行预测
    # inputs   : 游戏画面 obs
    def call(self, obs):
        # 根据层构建模型输出
        obs = self.ds1(obs)
        obs = self.ds2(obs)
        obs = self.ds3(obs)
        return obs

    # 反向传播 : 按照策略梯度算法进行更新参数
    # obs_list      : obs数组,shape :                 (None,6400)
    # act_list      : 经过obs数组预测得到的行为数组,    (None,1)
    # reward_list   : 行为获得的经过gama衰减的奖励,     (None,1)
    def learn(self, obs_list, act_list, reward_list):
        # 初始化学习率
        self.optimizer_ = optimizers.Adam(learning_rate=self.lr)
        obs_list = tf.cast(obs_list, dtype=tf.float32)
        reward_list = tf.cast(reward_list, dtype=tf.float32)
        # 求导过程,策略梯度算法的核心
        # 下列运算步骤可能还可以简化,调试没有达到提速的效果,有兴趣的自己试着再优化
        with tf.GradientTape() as tape:
            """
            表示预测的obs画面对应的action行为概率,其概率之和为1
            self(obs_list) 等价于 self.call(obs_list)
            案例 :
            predict_action_list :
            [
                [ 0.1 , 0.2 , 0.3 , 0.4],
                [ 0.1 , 0.5 , 0.2 , 0.2],
                ....
            ]

            """
            predict_action_list = self(obs_list)
            # 这一步 one_hot 是将预测到的action编码为热编码
            # 比如预测行为是 [ [1],... ] ,热编码 [ [0,1,0,0,0...],... ]
            one_hot_ = tf.one_hot(act_list, depth=self.act_dim, axis=1)
            """
                
            这一步是将热编码对应的行为的概率取出,其他行为概率抹除,然后求和得到结果
            比如  
                [0.1,  0.2,   0.3, ...]
            和热编码相乘得到 
                [0,           0.2,    0, ...]

            再进行求和,就得到 [ 0.2 ] 

            """
            predict_action_list *= one_hot_
            predict_action_list = tf.reduce_sum(predict_action_list, axis=1)
            
            """
            tf.math.log(0.1) 等价于 ln(0.1)
            假设 

                predict_action_list : [ 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9]

            那么的结果如下
            tf.math.log(predict_action_list) : 
                tf.Tensor(
                [ -2.3025851  -1.609438   -1.2039728  -0.9162907  -0.6931472  -0.5108256
                -0.35667497 -0.22314353 -0.10536055     ], shape=(9,), dtype=float32)

            通过上面的结果可以明白,这一步实际就是一个缩放的过程,使得概率越小的越小,概率越大的更加的大
            预测概率越小 log((obs,action)) = 0.1 的结果越小 -2.3025851
            预测概率越大 log((obs,action)) = 0.9 的结果越大 -0.10536055

            """
            log_ = tf.math.log(predict_action_list)
            """
            reward_list : 是经过gama处理以后的权重奖励,
            log_和reward_list相乘得到权重结果,再加一个负号,得到梯度上升的损失
            """
            loss = -log_ * reward_list
        gradient_ = tape.gradient(loss, self.trainable_variables)
        self.optimizer_.apply_gradients(zip(gradient_, self.trainable_variables))

