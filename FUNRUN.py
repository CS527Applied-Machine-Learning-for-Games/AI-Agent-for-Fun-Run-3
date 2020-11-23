import numpy as np
import cv2
from CNN import CNN
from PIL import Image
import pyautogui
import time



class FUNRUN(object):
    """
    This class acts as the intermediate "API" to the actual game. Double quotes API because we are not touching the
    game's actual code. It interacts with the game simply using screen-grab (input) and keypress simulation (output)
    using some clever python libraries.
    """
    pyautogui.FAILSAFE = False
    cnn_graph = CNN()
    reward = 5
    finish_rank = 4
    gameOver = False
    rank_templates=[None, cv2.imread("pos1.png",0), cv2.imread("pos2.png",0), cv2.imread("pos3.png",0), cv2.imread("pos4.png",0)]
    raceEndImage = cv2.imread('gameOver.png',0)
    stuckImage = cv2.imread('Stuck.png', 0)

    def __init__(self):
        self.reset()

    def findTemplate(self, image, template, threshold):
        method = cv2.TM_CCOEFF_NORMED
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        template = cv2.cvtColor(np.array(template), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(img_gray,template_gray,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)     
        h, w = template.shape[:-1]
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        loc = np.where(res >= threshold)
        if len(loc[0])>0:
            return True
        else:
            return False
        
    def _get_reward(self, action):
    	"""
    	return the current trank of our player
    	"""
    	img_rgb = pyautogui.screenshot()
    	# Convert it to grayscale
    	img_gray = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_BGR2GRAY)
    	for rank in range(1, 5):   
            # Read the template 
            template = self.rank_templates[rank]
            # Store width and height of template in w and h 
            w, h = template.shape[::-1] 
              
            # Perform match operations. 
            res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED) 
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)          
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            # Specify a threshold 
            threshold = 0.8
              
            # Store the coordinates of matched area in a numpy array 
            loc = np.where( res >= threshold)  
            if len(loc[0])>0:
                self.finish_rank = rank
                return self.reward - rank
        
    	return 5


    def _is_over(self, screen):
        img_rgb = cv2.resize(np.array(screen), (128, 128))
        cv2.imwrite("game.png", img_rgb)
        img_rgb = cv2.imread("game.png", 0)
        print("isGameOverCalled")
        x = 85
        for y in range(55,75):
            if img_rgb[x][y] != self.raceEndImage[x][y]:
                return False
        return True

    def _is_stuck(self, screen):
        img_rgb = cv2.resize(np.array(screen), (128, 128))
        cv2.imwrite("game.png", img_rgb)	
        img_rgb = cv2.imread("game.png", 0)
        print("isStuckCalled")
        x = 110
        for y in range(45,75):
            if img_rgb[x][y] != self.stuckImage[x][y]:
                return False
        return True
    
    def observe(self):
        print('\n\nobserve')
        self.gameOver = self._is_over(pyautogui.screenshot())
        if self.gameOver:
            time.sleep(6)
            try:
                pyautogui.locateOnScreen('home.png', confidence = 0.8)
                #print("home button located")
                pyautogui.press('h', presses=1)
                time.sleep(1)
                pyautogui.press('p', presses=1)
                time.sleep(11)
                #print("Race Started")
            except:
                time.sleep(25)
                #Ad playing
                return False
        elif self._is_stuck(pyautogui.screenshot()):
            #print("Stuck")
            self.gameOver = True
            self.finish_rank = 10
            pyautogui.press('p', presses=1)
            time.sleep(11)
            #print("Race Started")
        screen = cv2.resize(np.array(pyautogui.screenshot()), (1624, 750))
        state = self.cnn_graph.get_image_feature_map(screen)
        return state
        

    def act(self, action):
        pyautogui.press(' ', presses=1)
        if self.gameOver:
            self.gameOver = False
            finish_rank = 4
        display_action = ['jump', 'slide','slide','slide', 'slide', '3 Jump', '5 Jump', '7 Jump']
        print('action: ' + str(display_action[action]))
        keys_to_press = [['w'],['s'],['s'],['s'],['s'],[3],[5],[7]]
        if action >4:
        	num_presses = keys_to_press[action][0]
        	for i in range(num_presses):
        		pyautogui.press('w', presses =1)
        else:
	        for key in keys_to_press[action]:
	        	pyautogui.press(key, presses=1)
        self.reward = self._get_reward(action)
        return self.observe(), self.reward, self.gameOver, self.finish_rank

    def reset(self):
        return
