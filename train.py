import numpy as np
import gameMgr
import agent
import collections
#import matplotlib.pyplot as plt
import argparse,os,csv

def mean(x):
    return float(sum(x))/len(x)

showInterval = -1

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=10)
    parser.add_argument("--memory_limit",type=float,default=0.2)
    parser.add_argument("--learn_rate",type=float,default=1e-4)
    parser.add_argument("--eplen_bonus",type=float,default=0) # reward on the episode length
    parser.add_argument("--discount_rate",type=float,default=0.99)
    parser.add_argument("--replay_size",type=int,default=10000)
    parser.add_argument("--exploration",type=float,default=0.2)
    parser.add_argument("--soft_temp",type=float,default=1.0)
    parser.add_argument("--save_freq",type=int,default=100)
    parser.add_argument("--save_folder",type=str,default="model")
    parser.add_argument("--reload",type=str,default=None)
    parser.add_argument("--timeStep",type=int,default=40)
    args = parser.parse_args()

    gmm = gameMgr.tetris(20,10)
    gmm.setScore(drop=1,eplen_bonus=args.eplen_bonus,terminal=-1.)
    epoch  = 0
    write_epoch = 100
    reward_history     = collections.deque(maxlen=1000)
    rewardDrop_history = collections.deque(maxlen=1000)
    agt = agent.agent(gmm.getActionList(),gmm.getScreenSize(),gmm.getNextBlockSize(),nBatch=args.batch_size,timeStep=args.timeStep,learning_rate=args.learn_rate, discountRate=args.discount_rate, softTemp=args.soft_temp,saveFreq=args.save_freq, saveFolder=args.save_folder, memoryLimit=args.memory_limit)

    if args.reload : agt.load(args.reload)
    if not os.path.exists(args.save_folder): os.makedirs(args.save_freq)

    setFile = file(os.path.join(args.save_folder,"settings.dat"),"w")
    setFile.write(str(args))
    setFile.close()
    logFile = file(os.path.join(args.save_folder,"log.dat"),"w")
    logCSV  = csv.writer(logFile)
    logCSV.writerow(["epoch","length","last_rewardDrop","max_rewardDrop","mean_rewardDrop","last_reward","mean_reward"])
    while True:
        epoch  += 1
        gmm.startNewEpoch()
        state_tp1, reward, rewardDrop, _ = gmm.observe()
        terminal = False
        agt.clearExperience()
        reward_total = rewardDrop_total = 0.
        # start one experiment

        state_history = collections.deque(maxlen=args.timeStep)
        while not terminal:
            state_t = state_tp1
            state_history.append(state_t)
            action, value = agt.selectNextAction(state_history,T=args.soft_temp)
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
        logCSV.writerow([epoch,len(agt.experience),rewardDrop_history[-1],max(rewardDrop_history),mean(rewardDrop_history),reward_history[-1],mean(reward_history)])
        logFile.flush()
