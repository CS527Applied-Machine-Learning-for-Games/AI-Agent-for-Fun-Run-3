import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import sgd 
from matplotlib import pyplot as plt
from FUNRUN import FUNRUN
from train import train
from test import test


def baseline_model(grid_size, num_actions, hidden_size):
    # setting up the model with keras
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(grid_size,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.01), "mse")
    return model


def moving_average_diff(a, n=100):
    diff = np.diff(a)
    ret = np.cumsum(diff, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def load_model(count):
    # load json and create model
    json_file = open("final_model_epoch/model_epoch{}.json".format(count), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("final_model_epoch/model_epoch{}.h5".format(count))
    print("Loaded model from disk")
    loaded_model.compile(loss='mse', optimizer='sgd')
    return loaded_model



game = FUNRUN()
print("game object created")

print("Select mode:\n 1.Train\n 2.Test")
mode = int(input())
if mode == 1:
    # Train the model
    model = baseline_model(grid_size=2100, num_actions=8, hidden_size=512)
    epoch = int(input("Enter epochs to train"))
    hist, finish_ranks = train(game, model, epoch, verbose=1)
    print("Training done")
else:
    # Test the model
    model = load_model(83)
    epoch = int(input("Enter epochs to test"))  # Number of games to test
    hist, finish_ranks = test(game, model, epoch, verbose=1)

print("Race Wins: ", hist)
print("Finish Positions:", finish_ranks)
np.savetxt('win_history.txt', hist)
np.savetxt('win_rank.txt', finish_ranks)
