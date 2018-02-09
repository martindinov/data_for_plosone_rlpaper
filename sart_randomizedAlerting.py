import sys
import time
# import copy
import random
import subprocess
from psychopy import gui
from psychopy import visual, core, data, event
from time import sleep

import thread #for running server.serve_forever in a separate thread

import numpy as np
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding

import OSC

logger = logging.getLogger(__name__)

class SARTFeedbackEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.deltas = []
        self.thetas = []
        self.alphas = []
        self.betas = []
        self.gammas = []
        self.individualDeltas = []
        self.individualThetas = []
        self.individualAlphas = []
        self.individualBetas = []
        self.individualGammas = []
        self.eegType = None
        self.eeg = False
        self.eegReadings = []
        self.numReadingsToUseForEEGState = None
        self.reactionTimes = []
        self.serverForEEG = None
        self.win = None
        self.windowSize = 10
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=2, shape=(6,1))
        self.states = []
        self._seed()
        self.viewer = None
        self.state = None
        self.testing = False
        self.steps_beyond_done = None
        self.rewards = []
        self.actions = []
        self.correctResponses = []
        self.numTrials = 300
        self.numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.omitNum = 3
        self.fontSizes=[1.20, 1.80, 2.35, 2.50, 3.00]
        self.sart()

    def default_handler(self, addr, tags, stuff, source):
        return None
        
    def handler(self, addr, tags, data, client_address):
        #print "addr = ", addr
        #print "tags = ", tags
        if(math.isnan(data[2])):
            data[2] = 0
        if(math.isnan(data[3])):
            data[3] = 0

        avgReading = (data[2] + data[3])/2

        if(addr == "/muse/elements/delta_relative"):
            self.individualDeltas += [avgReading]
        elif(addr == "/muse/elements/theta_relative"):
            self.individualThetas += [avgReading]
        elif(addr == "/muse/elements/alpha_relative"):
            self.individualAlphas += [avgReading]
        elif(addr == "/muse/elements/beta_relative"):
            self.individualBetas += [avgReading]
        elif(addr == "/muse/elements/gamma_relative"):
            self.individualGammas += [avgReading]

    def useEEG(self, server="127.0.0.1", port=4445, eegType = "muse",
               numReadingsToUseForEEGState = 100):
        self.eeg = True
        self.eegType = eegType
        self.numReadingsToUseForEEGState = numReadingsToUseForEEGState
        self.serverForEEG = OSC.OSCServer((server, port))  # listen on localhost, port 4444
        if(self.eegType == "muse"):
            self.serverForEEG.addMsgHandler("default", self.default_handler) #overwrites default handler that otherwise complains if we don't have a handler registered for each message type (OSC address location)
            self.serverForEEG.addMsgHandler('/muse/elements/delta_relative', self.handler)
            self.serverForEEG.addMsgHandler('/muse/elements/theta_relative', self.handler)
            self.serverForEEG.addMsgHandler('/muse/elements/alpha_relative', self.handler)
            self.serverForEEG.addMsgHandler('/muse/elements/beta_relative', self.handler)
            self.serverForEEG.addMsgHandler('/muse/elements/gamma_relative', self.handler)
            thread.start_new_thread(self.serverForEEG.serve_forever, ())
            
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        #
        #    self.playAudioFFPlay('3000hz.wav')
        #    self.actions += [1]
        #else:
        #    self.actions += [0]


        if(self.np_random.uniform(0,1) <= 0.2):
            #play sound alert with small probability
            self.playAudioFFPlay('3000hz.wav')

        sleep(1)
            
        self.actions += [action]
        self.state = self.sart_trial(win=self.win)
        print "self.state ==============>>> ", self.state

        self.deltas += [np.mean(self.individualDeltas[-10:])]
        self.thetas += [np.mean(self.individualThetas[-10:])]
        self.alphas += [np.mean(self.individualAlphas[-10:])]
        self.betas += [np.mean(self.individualBetas[-10:])]
        self.gammas += [np.mean(self.individualGammas[-10:])]

        done = len(self.states) >= self.numTrials

        print "len(self.states) = ", len(self.states)
        if not done:
            if len(self.states) >= self.windowSize:
#                if np.mean(self.states[-(self.windowSize+1):-1]) >= (self.states[-1]):
#                    reward = 1
#                else:
#                    reward = -1
                reward = np.mean(self.states[-(self.windowSize+1):-1]) - self.states[-1]
            else:
                reward = 0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0

        self.rewards += [reward]
        return np.array([self.state, [self.deltas[-1]], [self.thetas[-1]], [self.alphas[-1]], [self.betas[-1]], [self.gammas[-1]]]), reward, done, {}

    def _reset(self):
        self.steps_beyond_done = None
        #self.states = [0]
        print "----------------Resetting----------------------"
        return np.array([[0],[0],[0],[0],[0],[0]])

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)


        if self.state is None: return None

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def playAudioFFPlay(self, audio_file_path):
        subprocess.call(["ffplay", "-nodisp", "-autoexit", audio_file_path])

    def sart(self, monitor="TestMonitor"):
        partInfo = self.part_info_gui()
        self.win = visual.Window(fullscr=True, color="black", units='cm',
                                 monitor=monitor)
        self.sart_init_inst(self.win, 3)
    
    def part_info_gui(self):
        info = gui.Dlg(title='SART')
        info.addText('Participant Info')
        info.addField('Part. Number: ')
        info.addField('Part. Gender: ', 
                      choices=["Please Select", "Male", "Female", "Other"])
        info.addField('Part. Age:  ')
        info.addField('Part. Year in School: ', 
                      choices=["Please Select", "Freshman", "Sophmore", "Junior", 
                           "Senior", "1st Year Graduate Student", 
                           "2nd Year Graduate Student", 
                           "3rd Year Graduate Student", 
                           "4th Year Graduate Student",
                           "5th Year Graduate Student", 
                           "6th Year Graduate Student"])
        info.addField('Do you have normal or corrected-to-normal vision?', 
                      choices=["Please Select", "Yes", "No"])
        info.addText('Experimenter Info')
        info.addField('DIS Initials:  ')
        info.show()
        if info.OK:
            infoData = info.data
        else:
            sys.exit()
            return infoData
        
    def sart_init_inst(self, win, omitNum):
        inst = visual.TextStim(win, text=("In this task, a series of numbers will" +
                                          " be presented to you.  For every" +
                                          " number that appears except for the" +
                                          " number " + str(omitNum) + ", you are" +
                                          " to press the space bar as quickly as" +
                                          " you can.  That is, if you see any" +
                                          " number but the number " +
                                          str(omitNum) + ", press the space" +
                                          " bar.  If you see the number " +
                                          str(omitNum) + ", do not press the" +
                                          " space bar or any other key.\n\n" +
                                          "Please give equal importance to both" +
                                          " accuracy and speed while doing this" + 
                                          " task.\n\nPress the b key when you" +
                                          " are ready to start."), 
                               color="white", height=0.7, pos=(0, 0))
        event.clearEvents()
        while 'b' not in event.getKeys():
            inst.draw()
            win.flip()
        
    def sart_prac_inst(win, omitNum):
        inst = visual.TextStim(win, text=("We will now do some practice trials " +
                                          "to familiarize you with the task.\n" +
                                          "\nRemember, press the space bar when" +
                                          " you see any number except for the " +
                                          " number " + str(omitNum) + ".\n\n" +
                                          "Press the b key to start the " +
                                          "practice."), 
                               color="white", height=0.7, pos=(0, 0))
        event.clearEvents()
        while 'b' not in event.getKeys():
            inst.draw()
            win.flip()
        
    def sart_act_task_inst(win):
        inst = visual.TextStim(win, text=("We will now start the actual task.\n" +
                                          "\nRemember, give equal importance to" +
                                          " both accuracy and speed while doing" +
                                          " this task.\n\nPress the b key to " +
                                          "start the actual task."), 
                               color="white", height=0.7, pos=(0, 0))
        event.clearEvents()
        while 'b' not in event.getKeys():
            inst.draw()
            win.flip()
            
    def sart_break_inst(win):
        inst = visual.TextStim(win, text=("You will now have a 60 second " +
                                          "break.  Please remain in your " +
                                          "seat during the break."),
                               color="white", height=0.7, pos=(0, 0))
        nbInst = visual.TextStim(win, text=("You will now do a new block of" +
                                            " trials.\n\nPress the b key " +
                                            "bar to begin."),
                                 color="white", height=0.7, pos=(0, 0))
        startTime = time.clock()
        while 1:
            eTime = time.clock() - startTime
            inst.draw()
            win.flip()
            if eTime > 60:
                break
            event.clearEvents()
            while 'b' not in event.getKeys():
                nbInst.draw()
                win.flip()

    def sart_block(win, fb, omitNum, reps, bNum, fixed):
        mouse = event.Mouse(visible=0)
        xStim = visual.TextStim(win, text="X", height=3.35, color="white", 
                                pos=(0, 0))
        circleStim = visual.Circle(win, radius=1.50, lineWidth=8,
                                   lineColor="white", pos=(0, -.2))
        numStim = visual.TextStim(win, font="Arial", color="white", pos=(0, 0))
        correctStim = visual.TextStim(win, text="CORRECT", color="green", 
                                      font="Arial", pos=(0, 0))
        incorrectStim = visual.TextStim(win, text="INCORRECT", color="red",
                                        font="Arial", pos=(0, 0))                                 


    def sart_trial(self, win):
        omitNum = self.omitNum
        fb=True
        clock=core.Clock()
        random.shuffle(self.fontSizes)
        random.shuffle(self.numbers)
        fontSize=self.fontSizes[0]
        number=self.numbers[0]

        mouse = event.Mouse(visible=0)
        xStim = visual.TextStim(win, text="X", height=3.35, color="white", 
                                pos=(0, 0))
        circleStim = visual.Circle(win, radius=1.50, lineWidth=8,
                                   lineColor="white", pos=(0, -.2))
        numStim = visual.TextStim(win, font="Arial", color="white", pos=(0, 0))
        correctStim = visual.TextStim(win, text="CORRECT", color="green", 
                                      font="Arial", pos=(0, 0))
        incorrectStim = visual.TextStim(win, text="INCORRECT", color="red",
                                        font="Arial", pos=(0, 0))                                 
        mouse.setVisible(0)
        respRt = "NA"
        numStim.setHeight(fontSize)
        numStim.setText(number)
        numStim.draw()
        event.clearEvents()
        clock.reset()
        stimStartTime = time.clock()
        win.flip()
        xStim.draw()
        circleStim.draw()
        waitTime = .25 - (time.clock() - stimStartTime)
        core.wait(waitTime, hogCPUperiod=waitTime)
        maskStartTime = time.clock()
        win.flip()
        waitTime = .90 - (time.clock() - maskStartTime)
        core.wait(waitTime, hogCPUperiod=waitTime)
        win.flip()
        allKeys = event.getKeys(timeStamped=clock)
        respAcc = 0
        if len(allKeys) != 0:
            respRt = allKeys[0][1]
        if len(allKeys) == 0:
            if omitNum == number:
                respAcc = 1
            else:
                respAcc = 0
        else:
            if omitNum == number:
                respAcc = 0
            else:
                respAcc = 1

        self.correctResponses += [respAcc]
        #print "correctResponses = ", self.correctResponses
                
        if fb == True:
            if respAcc == 0:
                incorrectStim.draw()
            else:
                correctStim.draw()
        stimStartTime = time.clock()
        win.flip()
        waitTime = .90 - (time.clock() - stimStartTime) 
        core.wait(waitTime, hogCPUperiod=waitTime)
        win.flip()
        endTime = time.clock()
        foo = 0



        if respRt == "NA":
            if respAcc == 1:
                reactionTime = np.mean(self.states)
            else:
                #incorrect response -> did not answer when they should have
                #penalize by treating this as very high RT of 2 seconds
                reactionTime = 2
        else: #subject answered
            if respAcc == 0: #incorrectly
                reactionTime = 2
            else: #correctly, in which case, reactionTime = respRt
                reactionTime = respRt

        self.states += [reactionTime]

        self.reactionTimes += [reactionTime]

        return [reactionTime]
