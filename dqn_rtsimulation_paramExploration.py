import dqn_rtsimulation

###define the parameter space exploration here###

maxNumLayers = 6
maxNeuronsPerLayer = 11
#11 instead of 1 as default here because with 11 we loop [0,10] and we divide gamma/10.0, so really we have 0,0.1,0.2,...,1.0
maxGamma = 11

#################################################

for numLayers in range(1,maxNumLayers): #1 to 6, excluding 6 (total of 5 
    print "Trying numLayers = ", numLayers
    for neuronsPerLayer in range(1, maxNeuronsPerLayer):
        print "Trying neuronsPerLayer = ", neuronsPerLayer
        for gamma in range(0,maxGamma):
            gamma = gamma/10.0
            print "Trying gamma = ", gamma
            for exploration in range(0,1):
                if exploration == 0:
                    explorationToUse = "epsilon"
                else:
                    explorationToUse = "boltzmann"

                for explorationParamTemp in range(0,11):

                    explorationParam = explorationParamTemp/float(10)
                    dqn_rt(nonLinearLayers=numLayers,
                                     neuronsPerLayer = neuronsPerLayer,
                                     epsilon = explorationParam,
                                     tau = explorationParam,
                                     exploration = explorationToUse,
                                     gamma = gamma)
                                     
                                     
