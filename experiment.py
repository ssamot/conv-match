import optparse
import sys
sys.setrecursionlimit(100000)

import numpy as np

from babi_helper import get_stories, vectorize_stories, tar, get_supporting_facts
from neuralnetworks import Logic
from config import task_path, tenK_task_path, tasks
from persistance import save_scores
from callbacks import LRScheduler, LearningRateAdapter
from utils import bcolors
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import os.path
import neuralnetworks

BATCH_SIZE = 64*2
EPOCHS = 50

ONLY_SUPPORTING = False

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


parser = optparse.OptionParser()
parser.add_option('--onlysup', action="store_true", dest="ONLY_SUPPORTING")
parser.add_option('--10K', action="store_true", dest="big_data")
parser.add_option('--memory', action="store_true", dest="memory")

parser.add_option("--hops", type="int", dest="hops")
parser.add_option("--runid", type="int", dest="runid")

(opts, args) = parser.parse_args()

runid = opts.runid

HOPS = opts.hops

if (opts.ONLY_SUPPORTING):
    ONLY_SUPPORTING = True

if (opts.big_data):
    tasks_full_path = [tenK_task_path + task for task in tasks]
else:
    tasks_full_path = [task_path + task for task in tasks]


nn = Logic()



rfilename = "Attention_" + "leakyrelu" +  "_" +  str(HOPS) + "_"
rheader = ""

if(opts.ONLY_SUPPORTING):
    rfilename+="onlysupp"
    rheader+="Only Supporting facts "
else:
    if(opts.memory):
        rfilename+="weak"
        rheader+="Weak Supervision "
    else:
        rfilename+="weak"
        rheader+="Weak Supervision "


if(opts.big_data):
    rfilename+="_10K"
    rheader+="10K "
else:
    rfilename+="_1K"
    rheader+="1K "

if(opts.memory):
    rfilename+="_mem"
    rheader+="Memory"
else:
    rfilename+=""
    rheader+=""



results = [rfilename,str(nn),rheader]
#print results
#exit()

print(bcolors.UNDERLINE + str(results) + bcolors.ENDC)

train = []
test = []

for i, task in enumerate(tasks_full_path):

    # if(i> 1):
    #     break

    print(bcolors.HEADER + "Loading task " + task + " ..." + bcolors.ENDC)
    train_data = tar.extractfile(task.format('train')).readlines()
    train.extend(get_stories(train_data, only_supporting=ONLY_SUPPORTING))

    test_data = tar.extractfile(task.format('test')).readlines()
    test.extend(get_stories(test_data, only_supporting=ONLY_SUPPORTING))



vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train + test)))




vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train + test)))

X, Xq, Y = vectorize_stories(train, word_idx, vocab_size, story_maxlen, query_maxlen)
tX, tXq, tY = vectorize_stories(test, word_idx, vocab_size, story_maxlen, query_maxlen)

print tY, Y

print('vocab = {}'.format(vocab))
print('X.shape = {}'.format(X.shape))
print('Xq.shape = {}'.format(Xq.shape))
print('Y.shape = {}'.format(Y.shape))
print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))



def multi_predict(mX, mXq, mY, fit):
            pts = []
            for j in range(100):
                pt = fit.predict([mX, mXq], batch_size=BATCH_SIZE).argmax(axis = -1)
                pts.append(pt)
                #pYs.append(tY)
                from scipy.stats import mode
                pts_merged = np.array(pts).transpose((1,0))
                pts_merged = mode(pts_merged, axis = -1)[0]
                pts_merged = pts_merged[:,0]

                acc = np.mean(mY.argmax(axis = -1) == pts_merged)
                print "real_acc", acc, j

            #acc = (xpred.argmax(axis = 1) == mY.argmax(axis = 1)).mean()

            return acc



if(opts.memory):

    np_filename_X = "./data/Xproc_" + str(i)
    np_filename_tX = "./data/tXproc_" + str(i)
    np_filename_ml = "./data/mlproc_" + str(i)
    if(opts.big_data):
        np_filename_X+="_10K.npy"
        np_filename_tX+="_10K.npy"
        np_filename_ml+="_10K.npy"
    else:
        np_filename_X+="_1K.npy"
        np_filename_tX+="_1K.npy"
        np_filename_ml+="1K.npy"



    if(not os.path.exists(np_filename_X + "i")):
        print("Padding...")
        from babi_helper import smartpadding, getmaxsentencelength
           # X = multiplex(X, word_idx)
        max_sentence_length = max(getmaxsentencelength(X, word_idx), getmaxsentencelength(tX, word_idx))
        X  = smartpadding(X, word_idx, max_sentence_length)
        tX = smartpadding(tX,  word_idx, max_sentence_length)

        X = X.astype(np.float32)
        Xq = Xq.astype(np.float32)
        Y = Y.astype(np.float32)
        tX = tX.astype(np.float32)
        tY = tY.astype(np.float32)
        np.save(np_filename_X, X)
        np.save(np_filename_tX, tX)
        #np.save(np_filename_ml, max_sentence_length)
    else:
        print("Caches found - loading model")
        tX = np.load(np_filename_tX)
        X = np.load(np_filename_X)
        max_sentence_length = int(np.load(np_filename_ml))
        print max_sentence_length




        # #def getmax(myX):
        # no_cores = 2
        # split_point = len(X)/no_cores
        # splits = [X[i*split_point:(i+1)*split_point] for i in range(no_cores+1)]
        # splits = [split for split in splits if split.shape[0]!=0]
        # print(len(splits))
        #
        # split_point_t = len(tX)/no_cores
        # splits_t = [tX[i*split_point_t:(i+1)*split_point_t] for i in range(no_cores+1)]
        # splits_t = [split for split in splits_t if split.shape[0]!=0]
        # print(len(splits_t))
        #
        #
        # def get_max(myX):
        #     return getmaxsentencelength(myX, word_idx)
        #
        # max_sentence_length = np.array((map(get_max, splits),map(get_max, splits_t))).max()
        # print("max_sentence_length", max_sentence_length)
        #
        # def get_smartpadding(myX):
        #     return smartpadding(myX, word_idx, max_sentence_length)
        #
        #
        #
        # X = map(get_smartpadding, splits)
        # tX = map(get_smartpadding, splits_t)
        # X = np.concatenate(X)
        # tX = np.concatenate(tX)



    #
    print("Compiling...")
    checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=0, save_best_only=True, monitor = "loss")
    rp = ReduceLROnPlateau(monitor = "loss", verbose = 0)

    attention = nn.distancenet(vocab_size, vocab_size, dropout = True, d_perc = 0.1, hop_depth = HOPS, type = "CCE", maxsize = max_sentence_length, shape = X.shape[1:], q_shape = Xq.shape[1] )
    #attention = nn.sequencialattention(vocab_size, vocab_size, dropout = True, d_perc = 0.2, hop_depth = 2, type = "CCE", maxsize = max_sentence_length)
    #attention = nn.softmaxattention(vocab_size, vocab_size, dropout = True, d_perc = 0.2, hop_depth = 1, type = "CCE", maxsize = max_sentence_length)



else:
    attention = nn.nomemory(vocab_size, vocab_size, dropout = True, d_perc = 0.1, type = "CCE")


try:
    sch = LRScheduler().schedule
    X = X.astype(np.float32)
    Xq = Xq.astype(np.float32)
    Y = Y.astype(np.float32)
    tX = tX.astype(np.float32)
    tY = tY.astype(np.float32)
    history = attention.fit([X, Xq], Y, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_data = [[tX, tXq], tY], callbacks=[checkpointer,rp])
except KeyboardInterrupt:
    print("Stoppping")


attention.load_weights("/tmp/weights.hdf5")

for i, task in enumerate(tasks_full_path):


    start = i*1000
    end = (i+1)*1000



    loss, acc = attention.evaluate([tX[start:end], tXq[start:end]], tY[start:end], batch_size=BATCH_SIZE)
    print loss, acc

    # acc = multi_predict(tX[start:end], tXq[start:end], tY[start:end], attention); loss = 0
    # print loss, acc

    print((bcolors.OKGREEN + 'Test loss / test accuracy = {:.5f} / {:.5f}' + bcolors.ENDC).format(loss, acc))
    score = "{:.5f} ".format(acc)

    results.append(score)

    print(bcolors.OKGREEN + str(results) + bcolors.ENDC)
    save_scores(results, runid)


#attention.save_weights("./weights/twohops16.weights")