Currently, the agent is a dumb agent randomly traversing a 3x3 grid parking lot. 
Its target parking spot is the top right square.
Each step, it returns its action value, immediate reward, cumulative reward and termination status.

Action Space:
The agent can perform one of four actions:
 1. Turn Left 90deg
 2. Turn Right 90deg
 3. Drive Forwards 1 Square
 4. Reverse Backwards 1 Square

Stop Conditions:
The simulation terminates when either one of two conditions are reached:
  1. Agent reaches target spot.

Reward System:
  1) If the car reaches the target parking spot (top right square), the agent receives +1 reward.
  2) If the car lands anywhere on the road (any middle row square), the agent receives 0 reward.
  3) If the car lands on any none road or target parking square (collides with other cars), the agent receives -3 reward.

Instructions to run code:
  Run the parkbot_v1.py file and the simulation will start. The simulation will end once either stop condition is reached.
