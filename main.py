from dqn import DQN
import cv2
import snake as game
import random
# for tensor
import numpy as np
import tensorflow as tf
import pygame

# hyperparameter
dis = 0.9

def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder = []
    src_vals = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = src_scope_name)
    dest_vals = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = dest_scope_name)

    for src_vals, dest_vals in zip(src_vals, dest_vals):
        op_holder.append(dest_vals.assign(src_vals.value()))
    return op_holder

def screen_handle(screen):
    procs_screen = cv2.cvtColor(cv2.resize(screen, (84, 84)), cv2.COLOR_BGR2GRAY)
    dummy, bin_screen = cv2.threshold(procs_screen, 1, 255, cv2.THRESH_BINARY)
    bin_screen = np.reshape(bin_screen, (84, 84, 1))

    return bin_screen

def replay_train(main_DQN, target_DQN, minibatch):
    x_stack = np.empty([1, 84, 84, 4])
    y_stack = np.empty([1, 4])

    for state, action, reward, next_state, done in minibatch:
        # 코드 다 수정하고 입력 사이즈 조절 부분 다시 보기
        state_temp = np.expand_dims(state, axis=0)
        next_state_temp = np.expand_dims(next_state, axis=0)

        Q = main_DQN.predict(state_temp)

        if done:
            Q[0, np.argmax(action)] = reward
        else:
            Q[0, np.argmax(action)] = reward + dis * np.max(target_DQN.predict(next_state_temp))

        y_stack = np.concatenate((y_stack, Q), axis = 0)
        x_stack = np.concatenate((x_stack, state_temp), axis=0)

        x_stack = x_stack[1:, :, :, :]
        y_stack = y_stack[1:, :]
    return main_DQN.update(x_stack, y_stack)



max_episodes = 10000


sess = tf.Session()

main_DQN = DQN(sess)
target_DQN =DQN(sess, name="target")

for episode in range(max_episodes):

    if episode == 0 :
        sess.run(tf.global_variables_initializer())

    e = 1. / ((episode / 10) + 1)
    done = False
    step_count = 0

    #state = env.reset()
    env = game.Snake()
    a_0 = np.array([1, 0, 0, 0])  # 상, 하, 좌, 우
    s_0, r_0, d = env.frameStep(a_0)
    s_0 = cv2.cvtColor(cv2.resize(s_0, (84, 84)), cv2.COLOR_BGR2GRAY)
    _, s_0 = cv2.threshold(s_0, 1, 255, cv2.THRESH_BINARY)

    main_DQN.initState(s_0)
    target_DQN.initState(s_0)


    while not done:

        # epsilon greedily
        action = np.zeros(main_DQN.action_size)

        if np.random.random() < e :
            idx = random.randrange(main_DQN.action_size)
            action[idx] = 1
        else:
            s_0 = np.expand_dims(main_DQN.s_t, axis=0)  # 1 84 84 4
            Q_pred = main_DQN.predict(s_0) #
            idx = np.argmax(Q_pred)
            action[idx] = 1


        screen_state, r_0, d = env.frameStep(action) # 200 200 3
        s_t1 = screen_handle(screen_state)  # s_t1: (84, 84, 1)

        epoch = main_DQN.addReplay(s_t1, action, r_0, d)

        if len(main_DQN.replay_buffer) > main_DQN.REPLAYMEM :
            main_DQN.replay_buffer.popleft()

        #s_0 = s_t1
        done = d
        step_count += 1

    print("episode: {}  step: {}".format(episode, step_count))

    # episode가 10번 지날 때 마다 학습을 시작한다.
    if episode > 100 :
        for _ in range(5):
            minibatch = random.sample(main_DQN.replay_buffer, 100)
            loss, _ = replay_train(main_DQN, target_DQN, minibatch)
            print("Loss: ", loss)
    # if episode % 100 == 0 :
    #     copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
    #     sess.run(copy_ops)
    #     print('target net copy')
