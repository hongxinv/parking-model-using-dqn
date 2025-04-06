Currently, the agent is a dumb agent randomly traversing a 3x3 grid parking lot. 
Its target parking spot is the top right square.

Action Space:
The agent can only move into directly adjacent squares (UP, DOWN, LEFT, RIGHT) and cannot move diagnolly.

Stop Conditions:
The simulation terminates when either one of two conditions are reached:
  1. Agent reaches target spot.
  2. Step limit is reached (Current step limit is 10 steps).

Reward System:
  1) If the car reaches the target parking spot (top right square), the agent receives +1 reward.
  2) If the car lands anywhere on the road (any middle row square), the agent receives 0 reward.
  3) If the car lands on any none road or target parking square (collides with other cars), the agent receives -3 reward.
