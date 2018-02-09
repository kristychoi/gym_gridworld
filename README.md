# gym_gridworld

Super small 4 x 4 gridworld, re-worked for compatibility in OpenAI gym environment.</br>
Original grid world implementation by sudarshanseshadri (https://github.com/sudarshanseshadri/gridworld)</br>

To install:
```
git clone https://github.com/kristychoi/gym_gridworld.git </b>
cd gym_gridworld
pip3 install -e .
```

To use (x,y) state space:
```
import gym
import gym_gridworld
env = gym.make('gym_gridworld-v0')
```

To use one-hot encoding of states (same grid world):
```
import gym
import gym_gridworld
env = gym.make('gym_onehotgrid-v0')
```

TODOs:
- implement rendering
- better initialization of environment
- code cleanup
