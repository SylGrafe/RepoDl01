# sygr0003 , UMU54907 , VT2019 , lab3_rl0 , simple RL (Reinforcment learning) demo 
# for comments read https://github.com/SylGrafe/RepoDl01/lab3/lab3sygr0003.pdf
# -*- coding: utf-8 -*-

# Modified version of rl_prorok.py 
import random
import gym
import sys
import matplotlib.pyplot as plt
import time
import datetime

dateFmt = "%d%m_%H%M%S"
startTime= datetime.datetime.now()
timeStample=startTime.strftime(dateFmt)

# nb of episodes  to run
EPISODES = 10000
# interval att which to check learning process
check_interval = int(EPISODES / 10)

EPSILON = 0.1    # random factor startvalue
EPSILON_MAX=0.15 # random factor minvalue
EPSILON_MIN=0.02 # random factor minvalue
myEpsilon=EPSILON # random factor 
epsilon_factor = 2 # use to change the value of epsilon

GAMMA = 0.9

# Original value :LR = 0.1
LEARNING_RATE = 0.02
#  when useRandom is true  and when qvalues are equals:
# choose action as random instead of choosing always left 
useRandom=False
useRandom=True

# nb of steps above which the run  is counted as  a good run
good_value = 180
# good run limit , value around which myepsilon will be  increase or diminished
goodCountLimit =  int (check_interval /6)
DISCRETE_STEPS = 10   # 10 discretization steps per state variable

# average of cumulative  rewards  , used to evaluate how well the model is learning
average_cumulative_reward = 0.0

def argmax(l):
  """ Return the index of the maximum element of a list
  """
  return max(enumerate(l), key=lambda x:x[1])[0]


def make_state(observation):
  """ Map a 4-dimensional state to a state index
  """
  low = [-4.8, -10., -0.41, -10.]
  high = [4.8, 10., 0.41, 10.]
  state = 0

  for i in range(4):
    # State variable, projected to the [0, 1] range
    state_variable = (observation[i] - low[i]) / (high[i] - low[i])

    # Discretize. A variable having a value of 0.53 will lead to the integer 5,
    # for instance.
    state_discrete = int(state_variable * DISCRETE_STEPS)
    state_discrete = max(0, state_discrete)
    state_discrete = min(DISCRETE_STEPS-1, state_discrete)

    state *= DISCRETE_STEPS
    state += state_discrete

  return state

def main():
  global average_cumulative_reward 
  good_count = 0
  sameQvalueCount = 0 
  global myEpsilon

  # Create the Gym environment (CartPole)
  env = gym.make('CartPole-v1')

  print('Action space is:', env.action_space)
  print('Observation space is:', env.observation_space)
  
  infoStr = "%s epsi , start%.2f [%.2f,%.2f] \n  lr: %.2f , random: %s \n good_run (val: %d , Limit:%d)"  % ( timeStample , 
  EPSILON , EPSILON_MIN , EPSILON_MAX, LEARNING_RATE , useRandom, good_value , goodCountLimit) 

  # print title for the check reports
  # print(" Check at  , av_value, good_count_ratio , sameQvalueCount " )
  print(" Check at  , av_value, good_count_ratio " )
 
  # Q-table for the discretized states, and two actions
  num_states = DISCRETE_STEPS ** 4
  qtable = [[0., 0.] for state in range(num_states)]

  average_reward = 0.
  #  use to plot  average cummulative reward for some runs
  xval=[]   # episode nb
  yval=[]  # avreage acc rewards
  zval=[]  # nb of good run in %
  
  # Loop over episodes
  for i in range(EPISODES):
    state = env.reset()
    state = make_state(state)

    terminate = False
    cumulative_reward = 0.0
    
    

    # Loop over time-steps
    # print("step,  cumulativ reward ")
    myCount=0
    while not terminate:
      myCount+=1
      # Compute what the greedy action for the current state is
      qvalues = qtable[state]
      # qtable[state][0] estimation of Value for action 0 (left) and state and "state"
      # qtable[state][1] estimation of Value for action 1 (right) 

      if (qvalues[0] ==  qvalues[1]) :
        sameQvalueCount += 1 
      greedy_action = argmax(qvalues)
      if (qvalues[0] ==  qvalues[1]) and useRandom:
        greedy_action = round(random.random())

      # Sometimes, the agent takes a random action, to explore the environment
      if random.random() < myEpsilon:
        action = random.randrange(2)
      else:
        # print (" greedy----> " , end="")
        action = greedy_action

        

      # Perform the action
      next_state, reward, terminate, info = env.step(action)  # info is ignored
      next_state = make_state(next_state)
      # print ("state , next_state : %d,%d"  %  (state ,next_state )  )
      # Show the simulated environment. A bit difficult to make it work.      
      # env.render()
      #print(' Reward:',reward)
      # Update the Q-Table
      td_error = reward + GAMMA * max(qtable[next_state]) - qtable[state][action]
      qtable[state][action] += LEARNING_RATE * td_error
      #print ("state , next_state : %d,%d"  %  (state ,next_state ) )
      #print ("qtable[%d][%d] : %s "  %  (state, action ,   qtable[state][action] ))
      # Update statistics
      cumulative_reward += reward
      if terminate:
      
      	average_reward += cumulative_reward
      	if cumulative_reward > good_value:
      	  #print("Good Job")
      	  good_count +=1
      prev_state = state
      
      state = next_state

      check_point = 0
      check_window = 1
      if ( i >= check_point and i <= check_point + check_window):
        # in state prev_state choose action and then update qvalues for 
        # prev_state [action]
        pass
        """
      	print ("action :%d --> qtable[%d] :  %.2f %.2f" % 
      	( action ,  prev_state , qtable[prev_state][0] , qtable[prev_state] [1]) )
      	"""
      elif i> check_point + check_window:
        # print ("check_point:%d , check_window :%d "  % (check_point , check_window ))
        #sys.exit(1)
        pass

    
    # Per-episode statistics
    if ((  (i+1) % check_interval)==0):
      mean_acc_reward=  average_reward/check_interval
      good_count_ratio=int (100*good_count/check_interval)
      """
      print("%d  , %.0f, %d , %d" % (
       (i+1), mean_acc_reward,good_count_ratio , sameQvalueCount), sep=',')
      """

      print("%d  , %.0f, %d " % (
       (i+1), mean_acc_reward,good_count_ratio ), sep=',')
       
      xval.append(i/1000)
      yval.append(mean_acc_reward)
      zval.append(good_count_ratio)

       
       
      average_reward = 0
      if (good_count > goodCountLimit and  myEpsilon > EPSILON_MIN  and i >= check_interval) :
        myEpsilon =  max( myEpsilon/epsilon_factor, EPSILON_MIN)
        print (" %d -----------------------------> myEpsilon:%.2f "  % ( i+1, myEpsilon))
      elif good_count < goodCountLimit and myEpsilon < EPSILON_MAX and i >= check_interval:
        myEpsilon = min(myEpsilon*epsilon_factor,EPSILON_MAX)
        print (" %d -----------------------------> myEpsilon:%.2f "  % ( i+1 , myEpsilon))
      good_count = 0
      sameQvalueCount = 0 
      
      
  # plot results
  info2= "epsilon:%.2f ,  lr: %.2f"   % (  EPSILON , LEARNING_RATE  ) 
  print (infoStr)
  fig=plt.figure(    "lab3_rl0" , figsize=(5, 4))

  theTitle =infoStr 
  fig.suptitle(theTitle, y=0.95,fontsize=10)

  #plt.subplots_adjust(bottom=0.2)
  plt.subplots_adjust(top=0.75)
  plt.subplots_adjust(bottom=0.10)

  ax = plt.subplot(2, 1, 1)
  #ax.set_title("mean_acc_rewards")
  # strange that you need to change hspace  and not wspace
  # to get  space enouth to see the tite of the second plot


  plt.xlabel('episodes/1000')
  plt.ylabel("mean Acc Rew")
  plt.plot( xval , yval ,  'bo'  )

  ax = plt.subplot(2, 1, 2)
  #ax.set_title("nb of good runs")
  plt.ylabel("goodRunRation")
  plt.plot( xval , zval ,  'r+'   )
  plt.xlabel('episodes/1000')

  #plt.ylabel("cummulative rewards")
  #fig.subplots_adjust(top=0.8)

  #title.set_y(1.05)
  #fig.subplots_adjust(hspace=0.5)

  #fig.subplots_adjust(vspace=0.5)


  plt.legend()
  figName="lab3_rl0" + timeStample
  plt.savefig (figName)
  plt.show(block=False)
  
      
#if __name__ == '__main__':
main()
