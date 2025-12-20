
# To do
1. identify S,A,O,T,Z,R,b - then see which of these i will need to change to fit my problem
    - 
1. understand the GUI
    - obstacle / object with WHITE hole = already located
    - red dot = belief distribution
1. 
1. check `agent.belief.py` <-- crucial for my last 5% of PF
1. 

# Note

Given: the position of obstacle (map layout), and list of target object

Objective: Find the (x, y) location of **n** target object


Solver: POUCT - not the original OO_POMCP

State:
- robot state
- env state: W x L grid world with
    - obstacle
    - target

Action
- Motion
    - default: scheme 3 (vx,vy)
    - 
- Look: Receive observation
- Find

Trasition model
- all action assumed to be deterministic
- **STATIC** environment

Observation
- return example: ```MosOOObservation({4: None, 6: (9, 3), 9: None, 11: None, 12: None})```
- sensor definition
    - laser: fan-shape
    - proxi
- param
    - epsilon
    - sigma

Belief
- prior
    - `init_robot_state`
-  

# POMDP Problem Structure
```
agent = pomdp_py.Agent(init_belief,
                        PolicyModel(),
                        TransitionModel(),
                        ObservationModel(obs_noise),
                        RewardModel())
env = pomdp_py.Environment(init_true_state,
                            TransitionModel(),
                            RewardModel())
```

# Robot capability
- pomdp-py stated: without considering rooms or topological graph (M_t).
    - does this mean there is no RRT for shortest path planning?
