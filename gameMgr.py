import os
import numpy as np
import matplotlib.pyplot as plt
import random,copy

AllBlocks = []
blocks = []
blocks.append([[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]])
blocks.append([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]])
blocks.append([[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]])
blocks.append([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]])
AllBlocks.append(blocks)
blocks = []
blocks.append([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]])
blocks.append([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]])
blocks.append([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]])
blocks.append([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]])
AllBlocks.append(blocks)
blocks = []
blocks.append([[0,1,0,0],[1,1,0,0],[1,0,0,0],[0,0,0,0]])
blocks.append([[1,1,0,0],[0,1,1,0],[0,0,0,0],[0,0,0,0]])
blocks.append([[0,1,0,0],[1,1,0,0],[1,0,0,0],[0,0,0,0]])
blocks.append([[1,1,0,0],[0,1,1,0],[0,0,0,0],[0,0,0,0]])
AllBlocks.append(blocks)
blocks = []
blocks.append([[1,0,0,0],[1,1,0,0],[0,1,0,0],[0,0,0,0]])
blocks.append([[0,1,1,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]])
blocks.append([[1,0,0,0],[1,1,0,0],[0,1,0,0],[0,0,0,0]])
blocks.append([[0,1,1,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]])
AllBlocks.append(blocks)
blocks = []
blocks.append([[1,1,1,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])
blocks.append([[0,1,0,0],[1,1,0,0],[0,1,0,0],[0,0,0,0]])
blocks.append([[0,1,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0]])
blocks.append([[1,0,0,0],[1,1,0,0],[1,0,0,0],[0,0,0,0]])
AllBlocks.append(blocks)
blocks = []
blocks.append([[1,1,1,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]])
blocks.append([[0,1,0,0],[0,1,0,0],[1,1,0,0],[0,0,0,0]])
blocks.append([[1,0,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0]])
blocks.append([[1,1,0,0],[1,0,0,0],[1,0,0,0],[0,0,0,0]])
AllBlocks.append(blocks)
blocks = []
blocks.append([[1,1,1,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]])
blocks.append([[1,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]])
blocks.append([[0,0,1,0],[1,1,1,0],[0,0,0,0],[0,0,0,0]])
blocks.append([[0,1,0,0],[0,1,0,0],[1,1,0,0],[0,0,0,0]])
AllBlocks.append(blocks)


class tetris:
    def __init__(self, n_rows, n_cols):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.screen_n_rows = n_rows
        self.screen_n_cols = n_cols
        self.possible_actions = (0, 1, 2, 3, 4)
        self.blockTypes = [ np.array(x,dtype=np.int32) for x in AllBlocks ] 
        self.reward_adrop = self.reward_epoch = 0
        self.iterCount = 0
        self.nextBlock = None

        # variables
        self.startNewEpoch()

        self.setScore()

    def setScore(self,drop=1.,eplen_bonus=0., terminal=-1.):
        self.score_drop = drop
        self.score_eplen= eplen_bonus
        self.score_terminal= terminal
        return
    def copy(self):
        t = tetris(self.screen_n_rows, self.screen_n_cols)
        t.setScore(self.score_drop,self.score_eplen,self.score_terminal)
        return t

    def getActionList(self):
        return self.possible_actions
    def getScreenSize(self):
        return (self.screen_n_rows, self.screen_n_cols)
    def getStateSize(self):
        return (self.screen_n_rows*self.screen_n_cols * 2 + 4*4)
    def getNextBlockSize(self):
        return (4*4)
    
    def drawBlock(self, blockIdx, blockPos, blockAng):
        temp = np.zeros((self.screen_n_rows, self.screen_n_cols), dtype=np.int32)
        if blockIdx==None:
            return temp
        for i in range(4):
            y = blockPos[0] + i
            if y<0 or y>=self.screen_n_rows: continue
            for j in range(4):
                x = blockPos[1] + j
                if x<0 or x>=self.screen_n_cols: continue
                temp[y,x] = self.blockTypes[blockIdx][blockAng][j,i]
        return temp

    def update(self, action):
        """
        action:
            0: do nothing
            1: move left
            2: move right
            3: turn left
            4: turn right
        """
        self.terminal = False

        # Move left/right
        blockIdx, blockPos, blockAng = self.blockIdx, copy.copy(self.blockPos), self.blockAng
        if   action == self.possible_actions[1]: blockPos[1] -= 1
        elif action == self.possible_actions[2]: blockPos[1] += 1
        elif action == self.possible_actions[3]: blockAng     = (blockAng+1)%4
        elif action == self.possible_actions[4]: blockAng     = (blockAng-1)%4
        temp = self.drawBlock(blockIdx, blockPos, blockAng)

        if temp.sum() == 4 and np.logical_and(temp,self.piles).sum()==0:
            self.blockPos, self.blockAng = blockPos, blockAng

        # Move down
        blockIdx, blockPos, blockAng = self.blockIdx, copy.copy(self.blockPos), self.blockAng
        blockPos[0] += 1
        temp = self.drawBlock(blockIdx, blockPos, blockAng)
        if temp.sum() == 4 and np.logical_and(temp,self.piles).sum()==0:
            self.blockPos, self.blockAng = blockPos, blockAng
        else:
            self.piles += self.drawBlock(self.blockIdx, self.blockPos, self.blockAng)
            self.blockIdx = None

        # Delete lines
        for i in range(self.screen_n_rows):
            if self.piles[i,:].sum() == self.screen_n_cols:
                self.reward_epoch += 1
                self.reward_adrop += 1
                self.piles[1:(i+1),:] = self.piles[:i,:]
                self.piles[0,:] = 0

        # loose check
        if self.piles[3,:].sum()>0:
            self.terminal = True

        if self.terminal:
            return False

        if self.blockIdx == None:
            self.reset()

        return True

    def observe(self):
        screen  = np.zeros((self.screen_n_rows, self.screen_n_cols), dtype=np.int32)
        piles = self.piles
        block = self.drawBlock(self.blockIdx,self.blockPos,self.blockAng)
        nextBlock = self.blockTypes[self.nextBlock[0]][self.nextBlock[1]]
        self.iterCount += 1
        reward_eplen = reward_drop = reward_terminal = 0.
        if self.terminal:
            reward_eplen = self.score_eplen * self.iterCount
            reward_terminal = -1.
        else:
            reward_drop  = self.reward_adrop * self.score_drop
            self.reward_adrop = 0
        screen  = np.logical_or(piles, block).astype(np.int32)
        screen += block.astype(np.int32)
        reward  = reward_drop + reward_eplen + reward_terminal
        return screen, reward, reward_drop, self.terminal

    def execute_action(self, action):
        return self.update(action)

    def reset(self):
        self.blockIdx = blockIdx = np.random.randint(0,len(self.blockTypes))
        self.blockAng = blockAng = np.random.randint(0,4)
        self.blockPos = blockPos = [0,np.random.randint(0,self.screen_n_cols-self.blockTypes[0].shape[1]+1)]
        if self.nextBlock==None:
            self.nextBlock = (blockIdx, blockAng, blockPos)
        else:
            self.blockIdx, self.blockAng, self.blockPos = self.nextBlock
            self.nextBlock  = (blockIdx, blockAng, blockPos)
        self.terminal = False
        return

    def startNewEpoch(self):
        self.iterCount = 0
        self.piles  = np.zeros((self.screen_n_rows, self.screen_n_cols), dtype=np.int32)
        self.reward_epoch = 0
        self.reset()
        return

if __name__=="__main__":
    t = tetris(16,10)
    fig = plt.figure(figsize=(10,5))
    fig.canvas.set_window_title("TeTris")
    epoch = 0
    while True:
        epoch += 1
        print("epoch=%d"%epoch)
        t.startNewEpoch()
        while True:
            action = random.randint(0,4)
            t.execute_action(action)
            temp, _, _ = t.observe()
            piles,block,_ = temp
            screen = piles+block
            img = plt.imshow(screen, interpolation="none", cmap="gray")
            fig.show()
            plt.pause(1e-10)

