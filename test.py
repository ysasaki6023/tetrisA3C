from __future__ import division

import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt

import gameMgr
import agent
import train


def init():
    img.set_array(state_t_1)
    plt.axis("off")
    return img,


def animate(step):
    global win, lose
    global agt
    global state_t_1, reward_t, terminal

    if terminal:
        gmm.startNewEpoch()
        #print()

        # for log
        #win += 1

        #print("WIN: {:03d}/{:03d} ({:.1f}%)".format(win, win + lose, 100 * win / (win + lose)))

    else:
        state_t = state_t_1

        # execute action in environment
        action_t, value = agt.selectMaxNextAction(state_t)
        print(value)
        gmm.execute_action(action_t)

    # observe environment
    state_t_1, reward_t, rewardDrop, terminal = gmm.observe()
    #screen = all_state_t_1[0] + all_state_t_1[1]
    #state_t_1 = train.flatten(all_state_t_1)

    # animate
    img.set_array(state_t_1)
    plt.axis("off")
    return img,


if __name__ == "__main__":
    # args
    frame_rate = 20
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-s", "--save", dest="save", action="store_true")
    parser.set_defaults(save=False)
    args = parser.parse_args()

    # environmet, agent
    #gmm = gameMgr.tetris(12,6)
    gmm = gameMgr.tetris(20,10)
    agt = agent.agent(gmm.getActionList(),gmm.getScreenSize(),gmm.getNextBlockSize(),n_batch=1,learning_rate=0, discountRate=0, saveFreq=0, saveFolder=None, memoryLimit=0.05)
    agt.InitSession(True)
    agt.load(args.model_path)

    # variables
    win, lose = 0, 0
    state_t_1, reward_t, rewardDrop, terminal = gmm.observe()
    #screen = all_state_t_1[0] + all_state_t_1[1]
    #state_t_1 = train.flatten(all_state_t_1)

    # animate
    fig = plt.figure(figsize=(gmm.getScreenSize()[0]/2,gmm.getScreenSize()[1]/2))
    fig.canvas.set_window_title("TeTris")
    img = plt.imshow(state_t_1, interpolation="none", cmap="gray")
    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=(1000 / frame_rate), blit=True)

    plt.show()
