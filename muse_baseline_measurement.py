import OSC
from math import isnan
import csv
from datetime import datetime
import calendar
import sys

deltas = []
thetas = []
alphas = []
betas = []
gammas = []
numValuesToRead = int(sys.argv[1])

def _default_handler(addr, tags, stuff, source):
    return None

def _handler(addr, tags, data, client_address):
    global deltas, thetas, alphas, betas, gammas
    #print "addr = ", addr
    #print "tags = ", tags
    if(isnan(data[2])):
        data[2] = 0
    if(isnan(data[3])):
        data[3] = 0

    avgReading = (data[2] + data[3])/2

    if(addr == "/muse/elements/delta_relative"):
        deltas += [avgReading]
    elif(addr == "/muse/elements/theta_relative"):
        thetas += [avgReading]
    elif(addr == "/muse/elements/alpha_relative"):
        alphas += [avgReading]
    elif(addr == "/muse/elements/beta_relative"):
        betas += [avgReading]
    elif(addr == "/muse/elements/gamma_relative"):
        gammas += [avgReading]
        print len(gammas)

    if(len(deltas) == numValuesToRead and len(thetas) == numValuesToRead
       and len(alphas) == numValuesToRead and len(betas) == numValuesToRead
       and len(gammas) == numValuesToRead):
        d = datetime.utcnow()
        unixtime = calendar.timegm(d.utctimetuple())
        ###save env.unwrapped.totalStates and env.unwrapped.actions as: [state,action] pairs
        with open('muse_baseline_measurement_' + str(unixtime) + '.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerows([deltas[1:numValuesToRead], thetas[1:numValuesToRead], alphas[1:numValuesToRead], betas[1:numValuesToRead], gammas[1:numValuesToRead]])
            print "Done writing out measurements to a csv file..."
            s.server_close()

try:
    if __name__ == "__main__":
        s = OSC.OSCServer(('127.0.0.1', 4445))  # listen on localhost, port 57120
        s.addMsgHandler("default", _default_handler)
        s.addMsgHandler('/muse/elements/delta_relative', _handler)
        s.addMsgHandler('/muse/elements/theta_relative', _handler)
        s.addMsgHandler('/muse/elements/alpha_relative', _handler)
        s.addMsgHandler('/muse/elements/beta_relative', _handler)
        s.addMsgHandler('/muse/elements/gamma_relative', _handler)
        s.serve_forever()
except Exception:
    print "Interrupted the server after writing to a csv file"
