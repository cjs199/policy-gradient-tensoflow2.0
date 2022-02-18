import time
import numpy as np
import cjs_util
import gym
from pg_ import *

# 解决 tensorflow2 加载模型时报错的问题
# pip install keras==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 游戏环境完善
# pip install gym -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install ale-py  -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install gym[accept-rom-license] -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装后会报一些错,但测试已经可以运行
# pip install gym[all] -i https://pypi.tuna.tsinghua.edu.cn/simple


# 需要放在tensorflow调用前 , 动态显存,不要全部占用
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

total_for = 20000

resize = 0.5
o_dim = int(160 * resize)
t_dim = o_dim
dim = o_dim * t_dim
img_index = 0
random_index = 0

# 创建游戏
game = gym.make('Pong-v0')  # 0 up, 1 right, 2 down, 3 left
act_dim = game.action_space.n

# 缓存
# 图像
obs_list = []
# 行为
act_list = []
# 奖励
reward_list = []

# 初始化加权分数
calc_reward = -1.3128734

# 创建并打印模型
train_model = PG(act_dim, 5e-4)
train_model.build(input_shape=[None, dim])
train_model.summary()

# 根据画面生成对应的行为
def send_act(obs):
    global random_index
    if random_index > 0:
        random_index -= 1
        if not random_index > 0:
            print('随机结束')
        predict = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
        return np.random.choice(len(predict), p=predict)
    # train_model(obs_list) 等价于 train_model.call(obs_list)
    predict = train_model(np.array([obs]))
    predict = predict[0]
    # predict 得到的结果概率之和,并不是完全等于 1 ,导致报错,所以在工具类中包装一个方法,重新分布,让概率和为1
    return cjs_util.choice(predict.numpy())

# 运行游戏,收集数据
def run():
    # 记录迭代了多少步
    total_reward = 0
    total_step = 0
    # 初始化游戏
    obs = game.reset()
    obs = change_obs(obs)
    # 蒙特卡洛法,一方获得21分时才结束,将本轮所有数据收集起来进行学习
    for i in range(20000):
        total_step += 1
        act = send_act(obs)
        # 发送游戏行为,并获取响应
        next_obs, reward, done, _ = game.step(act)
        # 如果游戏已经结束,截断
        if done:
            break
        # 缓存数据
        obs_list.append(obs)
        act_list.append(act)
        reward_list.append(reward)
        # 统计总分数
        total_reward += reward
        # 显示游戏画面
        game.render()
        # 更新obs,作为下一次预测行为的依据
        obs = change_obs(next_obs)

    # 加权分数,旧的累计分数权重0.99,新的训练分数权重0.01,这样得到的分数更加稳定
    global calc_reward
    calc_reward = calc_reward * 0.99 + total_reward * 0.01

    # 将训练结果追加到日志中
    cjs_util.appand_log('./train.log',
                        '总共运行步数: ' + cjs_util.add_append_str(str(total_step), 0, 4, ' ') +
                        ' , 分数: ' + cjs_util.add_append_str(str(total_reward), 0, 5, ' ') +
                        ' , 加权分数: ' + cjs_util.add_append_str(str(calc_reward), 0, 10) + '\r')

# 转换obs为可用的
def change_obs(image):
    """ 预处理 210x160x3 uint8 frame into 6400 (80x80) 1维 float vector """
    image = image[35:195]  # 裁剪
    image = image[::int(1 / resize), ::int(1 / resize), 0]  # 下采样，缩放
    image[image == 144] = 0  # 擦除背景 (background type 1)
    image[image == 109] = 0  # 擦除背景 (background type 2)
    image[image != 0] = 1  # 转为灰度图，除了黑色外其他都是白色
    return image.astype(np.float64).ravel()

# gamma衰减奖励,并进行标准化
def gamma_reward(reward_list, gamma=0.99):
    """calculate discounted reward"""
    reward_arr = np.array(reward_list)
    for i in range(len(reward_arr) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_arr[i] += gamma * reward_arr[i + 1]
    # 标准化奖励
    reward_arr -= np.mean(reward_arr)
    reward_arr /= np.std(reward_arr)
    return reward_arr

# 学习
def learn():
    gr = gamma_reward(reward_list)
    train_model.learn(obs_list, act_list, gr)

# 加载模型,用来测试
def load_model():
    return tf.keras.models.load_model('./策略梯度/pg')

# 加载模型权重
def load_weights(model):
    model.load_weights('./策略梯度/pg_weights')


# 保存模型
def save_weights(model):
    # 保存权重,下一次启动时加载此模型用来训练
    model.save_weights('./策略梯度/pg_weights')
    # 保存模型,此模型测试使用,不能用来训练
    model.save('./策略梯度/pg')

# 训练模型
def tarin_mode():

    # 加载模型权重
    load_weights(train_model)

    for for_index in range(total_for):

        # 积累训练数据
        run()

        # 开始学习
        learn()

        # 没10次保存下模型
        if for_index % 10 == 0:
            save_weights(train_model)

        # 清空缓存的所有数据
        # 画面
        obs_list.clear()

        # 获取 act
        act_list.clear()

        # 获取 reward
        reward_list.clear()

# 测试模型
def test_mode():

    # 加载模型权重
    global train_model
    train_model = load_model()

    # 初始化游戏
    obs = game.reset()
    obs = change_obs(obs)
    # 蒙特卡洛法,一方获得21分时才结束,将本轮所有数据收集起来进行学习
    for i in range(20000):
        act = send_act(obs)
        # 发送游戏行为,并获取响应
        next_obs, reward, done, _ = game.step(act)
        # 睡眠1s钟,避免运行过快
        # 显示游戏画面
        game.render()
        time.sleep(0.005)
        # 如果游戏已经结束,截断
        if done:
            break
        # 更新obs,作为下一次预测行为的依据
        obs = change_obs(next_obs)



if __name__ == '__main__':
    
    # 训练模型
    tarin_mode()

    # 测试模型,不训练更新模型
    # test_mode()