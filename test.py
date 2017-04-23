from __future__ import division

import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import gameMgr
import agent
import train
import collections


def init():
    img.set_array(state_tp1)
    plt.axis("off")
    return img,


def animate(step):
    global win, lose, totCount, doSave
    global agt
    global state_tp1, reward_t, terminal
    global softTemp, timeStep
    global lstm_state
    global state_history
    global score, length, avg_score, avg_length

    if doSave:
        if totCount<0: return
        totCount -= 1

    if terminal:
        avg_score.append(score)
        avg_length.append(length)
        print "game over: length=%d, score=%d    avg(length)=%.1f avg(score)=%.3f"%(length,score,np.mean(avg_length),np.mean(avg_score))
        score = 0
        length = 0
        gmm.startNewEpoch()

    else:
        state_t = state_tp1
        state_history.append(state_t)

        # execute action in environment
        action_t, value, lstm_state = agt.selectNextMaxAction([state_t],prev_lstm_state=lstm_state)
        #action_t, value, lstm_state = agt.selectNextMaxAction(state_history,prev_lstm_state=lstm_state)
        #action_t, value = agt.selectNextAction(state_history,T=1e-10)
        #print(value)
        gmm.execute_action(action_t)

    # observe environment
    state_tp1, reward_t, rewardDrop, terminal = gmm.observe()
    length += 1
    score  += rewardDrop

    # animate
    img.set_array(state_tp1)
    plt.axis("off")

    return img,


if __name__ == "__main__":
    # args
    frame_rate = 20
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-s", "--save", dest="save",default=False)
    parser.add_argument("-n", "--logLength", dest="logLength",default=1000,type=int)
    #parser.set_defaults(save=False)
    args = parser.parse_args()

    # parse
    dirpath = os.path.dirname(args.model_path)
    fcont = file(os.path.join(dirpath,"settings.dat"),"r")
    cont  = fcont.readline()
    cont  = cont.replace("Namespace","").replace("=",":").replace("(","{'").replace(")","}").replace(":","':").replace(", ",", '")
    cont  = eval(cont)
    print(cont)

    # environmet, agent
    lstm_state = None
    score, length = 0,0
    avg_score, avg_length = collections.deque(maxlen=1000), collections.deque(maxlen=1000)
    #softTemp = 1e-20
    softTemp = cont["soft_temp"]
    timeStep = cont["timeStep"]
    gmm = gameMgr.tetris(20,10)
    agt = agent.agent(gmm.getActionList(),gmm.getScreenSize(),gmm.getNextBlockSize(),nBatch=1,timeStep=timeStep,learning_rate=0, discountRate=0, saveFreq=0, softTemp=softTemp, saveFolder=None, memoryLimit=0.05)
    agt.load(args.model_path)

    # variables
    win, lose = 0, 0
    state_tp1, reward_t, rewardDrop, terminal = gmm.observe()
    totCount = args.logLength
    doSave = args.save
    state_history = collections.deque(maxlen=timeStep)

    # animate
    fig = plt.figure(figsize=(gmm.getScreenSize()[0]/2,gmm.getScreenSize()[1]/2))
    fig.canvas.set_window_title("TeTris")
    img = plt.imshow(state_tp1, interpolation="none", cmap="gray")
    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=(1000 / frame_rate), blit=True)

    if args.save:
        ani = animation.FuncAnimation(fig, animate, init_func=init, interval=(1000 / frame_rate), blit=True,save_count=args.logLength)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Yuichi Sasaki',title="tetris A3C"))
        ani.save(args.save+".mp4", writer=writer)

    plt.show()
