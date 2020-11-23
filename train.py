import numpy as np
import time
from getkeys import key_check
from ExperienceReplay import ExperienceReplay

# parameters
num_actions = 8
max_memory = 1000  # Maximum number of experiences we are storing
batch_size = 4  # Number of experiences we use for training per batch

exp_replay = ExperienceReplay(max_memory=max_memory)


def save_model(model, count):
    # serialize model to JSON
    model_json = model.to_json()
    with open("final_model_epoch/demo_model_epoch{}.json".format(count), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("final_model_epoch/demo_model_epoch{}.h5".format(count))
    # print("Saved model to disk")


def train(game, model, epochs, verbose=1):
    # Reseting the win counter
    win_cnt = 0
    # We want to keep track of the progress of the AI over time, so we save its win count history
    win_hist = []
    ranks = []
    # Epochs is the number of games we play
    for e in range(epochs):
        loss = 0.
        epsilon = 0.4
        # Resetting the game
        game.reset()
        game_over = False
        # get tensorflow running first to acquire cudnn handle
        input_t = game.observe()
        if e == 0:
            paused = True
            print('Training is paused. Press X once game is loaded and is ready to be played.')
        else:
            paused = False
        while not game_over:
            if not paused:
                # The learner is acting on the last observed game screen
                # input_t is a vector containing representing the game screen
                input_tm1 = input_t

                """
                We want to avoid that the learner settles on a local minimum.
                Imagine you are eating in an exotic restaurant. After some experimentation you find 
                that Penang Curry with fried Tempeh tastes well. From this day on, you are settled, and the only Asian 
                food you are eating is Penang Curry. How can your friends convince you that there is better Asian food?
                It's simple: Sometimes, they just don't let you choose but order something random from the menu.
                Maybe you'll like it.
                The chance that your friends order for you is epsilon
                """
                if np.random.rand() <= epsilon:
                    # Eat something random from the menu
                    action = int(np.random.randint(0, num_actions, size=1))
                    print('random action')
                else:
                    # Choose yourself
                    # q contains the expected rewards for the actions
                    q = model.predict(input_tm1)
                    # We pick the action with the highest expected reward
                    action = np.argmax(q[0])

                # apply action, get rewards and new state
                input_t, reward, game_over, finish_rank = game.act(action)
                if game_over:
                    if finish_rank == 10:
                        reward = -10
                        ranks.append(-1)
                    else:
                        win_cnt += 1
                        reward += (5 - finish_rank)*3
                        ranks.append(finish_rank) 
                """
                The experiences < s, a, r, s’ > we make during gameplay are our training data.
                Here we first save the last experience, and then load a batch of experiences to train our model
                """

                # store experience
                exp_replay.remember([input_tm1, action, reward, input_t], game_over)

                # Load batch of experiences
                inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

                # train model on experiences
                batch_loss = model.train_on_batch(inputs, targets)

                # print(loss)
                loss += batch_loss

            # menu control
            keys = key_check()
            if 'X' in keys:
                if paused:
                    paused = False
                    print('unpaused!')
                    time.sleep(1)
                else:
                    print('Pausing!')
                    paused = True
                    time.sleep(1)
            elif 'Q' in keys:
                print('Quitting!')
                return

        if verbose > 0:
            print("Epoch {:03d}/{:03d} | Loss {:.4f} | Completed Games {}".format(e+1, epochs, loss, win_cnt))
        save_model(model, e)
        win_hist.append(win_cnt)
    return win_hist, ranks
