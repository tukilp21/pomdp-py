# To do
1. identify S,A,O,T,Z,R,b - then see which of these i will need to change to fit my problem
    - 
1. understand the GUI
    - obstacle / object with WHITE hole = already located
    - red dot = belief distribution
 
1. check `agent.belief.py` <-- crucial for my last 5% of PF
    - check belief update and how it called in **solver**
    - object-oriented belief: one distribution (across the grid OR over/exclue some prior) per object

1. `observation_model_py` looks complicated 
    - my intuition: to model/define, especially on in-domain simplifying assumption, so that belief update can be simply *bayesian update* (as stated in `agent/belief.py`)

1. play around with `class StaticObjectTransitionModel(pomdp_py.TransitionModel)`

# Bug 

## To fix / add
- sensor define with *occlusion True* but still looks through wall (tested on world 1, the agent chooses *look* east from the start). from Coplit:
    - Obstacles are NOT included in sensor observation?? check `env.env`
    - `models/components/sensor.py` line 224-251, 211-221
    - 
- `agent_has_map=False` <-- help the agent avoid collision, but no Penalty applied yet

## Fixed
- `_build_beam_map`, `observe`

# Note

## The problem
Given: the position of obstacle (map layout), and list of target object (fixed set to have belief over)

Objective: Find the (x, y) location of **n** target object

Solver: POUCT - not the original OO_POMCP

## POMDP Problem Structure

### State space is W x L grid world, where s_t defined object-oriented state
s = {robot_state, obj_1, obj_2, obj_3, obj_n}


### Action
- Motion
    - by default: scheme 3 (vx,vy)
- Look: Receive observation
- Find

### Transition model
- all action assumed to be deterministic
- **STATIC** environment

### Observation
Defined in `env.env.py`

- return example: ```MosOOObservation({4: None, 6: (9, 3), 9: None, 11: None, 12: None})```
- sensor definition
    - laser: fan-shape
    - proxi
- param
    - epsilon
    - sigma

## Belief (`agent/belief.py`)
- `initialize_belief()`
    -   A mapping {(objid|robot_id) -> {(x,y) -> [0,1]}}
- prior 
    - *line 62, 80* check `class MosOOPOMDP(pomdp_py.OOPOMDP)` in `problem.py`, 

- `belief_update()` in `problem.py`
    - in practice: intractable to obtain s'
    - simplifying assumption: object is contained within one pixel (or voxel), so Observation ~ Labelling

- **`solve()`**
    - **Line 209** select solver based on Belief repr.
    - **Line 276** call `belief_update()`

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
