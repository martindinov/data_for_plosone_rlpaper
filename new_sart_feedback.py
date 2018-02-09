import expyriment
import random
import time
import numpy as np

numbers = [1,2,3,4,5,6,7,8,9]

exp = expyriment.design.Experiment(name="RL-SART")
expyriment.control.initialize(exp)

block = expyriment.design.Block(name="A name for the first block")

for trialNumber in range(0,200):
    trial = expyriment.design.Trial()
    maskStim = expyriment.stimuli.Circle(70, colour=(255,255,255))
    random.shuffle(numbers)
    numberStim = expyriment.stimuli.TextLine(text=str(numbers[0]), text_bold=True, text_size=80, text_colour=(255,255,255))
    correctStim = expyriment.stimuli.TextLine(text="CORRECT", text_size=60, text_colour=(0,255,0))
    incorrectStim = expyriment.stimuli.TextLine(text="INCORRECT", text_size=60, text_colour=(255,0,0))
    maskStim.preload()
    numberStim.preload()
    correctStim.preload()
    incorrectStim.preload()
    trial.add_stimulus(maskStim)
    trial.add_stimulus(numberStim)
    trial.add_stimulus(correctStim)
    trial.add_stimulus(incorrectStim)
    trial.set_factor(str(trialNumber), numbers[0])
    block.add_trial(trial)

exp.add_block(block)


print "block trials = ", block.trials[0]


expyriment.control.start()

for block in exp.blocks:
    trialNumber = -1
    for trial in block.trials:
        trialNumber += 1
        trial.stimuli[0].present()
        exp.clock.wait(np.random.uniform(200,700))
        trial.stimuli[1].present()
        key, rt = exp.keyboard.wait([expyriment.misc.constants.K_SPACE], duration=2000)
        print "rt = ", rt
        print "key = ", key
        print "trial number = ", trial.get_factor(str(trialNumber))
        
        correct = None
        if(trial.get_factor(str(trialNumber)) == 3):
            if(key is None):
                correct = True
            else:
                correct = False
        else:
            if(key is None):
                correct = False
            else:
                correct = True
                
        if correct:
            trial.stimuli[2].present()
        else:
            trial.stimuli[3].present()

        exp.clock.wait(750)


expyriment.control.end()
