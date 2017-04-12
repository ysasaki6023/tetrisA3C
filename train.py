import numpy as np
import gameMgr
import agent
import collections
import threading, signal
import argparse,os,csv
from rmsprop_applier import RMSPropApplier

def mean(x):
    return float(sum(x))/len(x)

showInterval = -1

def train_function(parallel_index,agt,gmm):
    global stopFlag
    global epoch
    global global_agt
    while True:
        if stopFlag: break
        agt.sess.run( agt.sync )
        epoch  += 1
        gmm.startNewEpoch()
        state_tp1, reward, rewardDrop, _ = gmm.observe()
        terminal = False
        agt.clearExperience()
        reward_total = rewardDrop_total = 0.
        # start one experiment
        while not terminal:
            state_t = state_tp1
            action, value = agt.selectNextAction(state_t)
            gmm.execute_action(action)
            state_tp1, reward, rewardDrop, terminal = gmm.observe()
            agt.storeExperience(state_t, action, value, state_tp1, reward, terminal)
            reward_total += reward
            rewardDrop_total += rewardDrop
        reward_history.append(reward_total)
        rewardDrop_history.append(rewardDrop_total)
        # start training
        summary = agt.trainFromExperience(addSummary={"rewardDrop":rewardDrop, "length":len(agt.experience), "rewardDropAvg":mean(rewardDrop_history), "rewardAvg":mean(reward_history)})
        if epoch%write_epoch==0:
            agt.writer.add_summary(summary,epoch)
        if epoch%args.save_freq==0:
            agt.saver.save(agt.sess,os.path.join(args.save_folder,"model.ckpt"),epoch)

        print "epoch=%5d"%epoch,"length=%3d"%len(agt.experience),"rewardDrop=%2d"%rewardDrop_history[-1],"rewardDrop_max=%2d"%max(rewardDrop_history),"rewardDrop_avg=%.3f"%mean(rewardDrop_history),"reward=%.3f"%reward_history[-1],"reward_avg=%.3f"%mean(reward_history)
    return

def signal_handler(signal, frame):
  global stopFlag
  print('You pressed Ctrl+C!')
  stopFlag = True
  return
  

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=256)
    parser.add_argument("--memory_limit",type=float,default=0.15)
    parser.add_argument("--learn_rate",type=float,default=1e-3)
    parser.add_argument("--eplen_bonus",type=float,default=0) # reward on the episode length
    parser.add_argument("--discount_rate",type=float,default=0.99)
    parser.add_argument("--replay_size",type=int,default=10000)
    parser.add_argument("--exploration",type=float,default=0.2)
    parser.add_argument("--save_freq",type=int,default=100)
    parser.add_argument("--save_folder",type=str,default="model")
    parser.add_argument("--reload",type=str,default=None)
    parser.add_argument("--nworkers",type=int,default=1)
    args = parser.parse_args()

    if not os.path.exists(args.save_folder): os.makedirs(args.save_folder)
    setFile = file(os.path.join(args.save_folder,"settings.dat"),"w")
    setFile.write(str(args))
    setFile.close()
    epoch  = 0
    write_epoch = 100
    reward_history     = collections.deque(maxlen=1000)
    rewardDrop_history = collections.deque(maxlen=1000)
    global_gmm = gameMgr.tetris(20,10)
    global_gmm.setScore(drop=1,eplen_bonus=args.eplen_bonus,terminal=-1.)
    global_agt = agent.agent(global_gmm.getActionList(),global_gmm.getScreenSize(),global_gmm.getNextBlockSize(),n_batch=args.batch_size,learning_rate=args.learn_rate,
            discountRate=args.discount_rate, saveFreq=args.save_freq, saveFolder=args.save_folder, memoryLimit=args.memory_limit, device="/gpu:0")

    grad_applier = RMSPropApplier(learning_rate = args.learn_rate, device="/gpu:0")

    if args.reload : agt.load(args.reload)

    stopFlag = False
    # build threads
    threadList = []
    for i in range(args.nworkers):
        gmm = global_gmm.copy()
        local_agt = agent.agent(gmm.getActionList(),gmm.getScreenSize(),gmm.getNextBlockSize(),n_batch=args.batch_size,learning_rate=args.learn_rate,
                                discountRate=args.discount_rate, saveFreq=args.save_freq, saveFolder=args.save_folder, memoryLimit=args.memory_limit)
        local_agt.copySettings(global_agt,grad_applier,i)
        threadList.append(threading.Thread(target=train_function, args=(i,local_agt,gmm,)))
    global_agt.InitSession(init=True)
    for i in range(args.nworkers):
        local_agt.sess = global_agt.sess
    # set signal handlers
    signal.signal(signal.SIGINT, signal_handler)

    # execute threads
    for t in threadList:
        t.start()

    # termination
    print('Press Ctrl+C to stop')
    signal.pause()
    print('Waiting for threads to back')
    for t in threadList:
        t.join()
    #print('Now saving data. Please wait')
    #saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)

