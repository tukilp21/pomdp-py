# My note 

## To do
- sensor define with *occlusion True* but still looks through wall (tested on world 1, the agent chooses *look* east from the start). from Coplit:
    - `models/components/sensor.py` line 224-251, 211-221

- `agent_has_map`: for action sampling only - have not used to make belief update or planning better
- observation model, `sample_zi()`

## Fixed
- `_build_beam_map`, `observe`

## Make it better
- `vis` >>> draw_belief >>> utils.lighter: is the idea is to illustrate the belief as a heatmap?
- still see through wall regardless of occlusion `model/components/sensor.py` check function `observe`. 
    - testing look behaviour:
    1. trye increase *solver's depth, planning_time, exploration_const*
    1. reduce negative reward for look
    - no hope... seem like whenever i set occlusion to True, it affect the *planning* which output almost NO *look* action. even when the agent goes into free space.
    - *laser_sensor* also the same
- for U shape occluded world3, POUCT seem to get stuck most of the time. Wonder if **POMCP** (not yet implemented properly) would perform better?

# Overview

## The problem
Given: 
- list of obstacble object (can also give the position)
- list of target object (fixed set to have belief over)

Objective: Find the (x, y) location of target object

Solver: POUCT (not the original OO-POMCP) with Histogram as belief representation

## POMDP Problem Structure

### Domain
#### State space is W x L grid world, where s_t defined object-oriented state
s = {robot_state, obj_1, obj_2, obj_3, obj_n}
- each object has attributes: id, type, pose (x,y), found (T/F)

#### Action
- Motion
    - by default: scheme 3 (vx,vy)
- Look: Receive observation
- Find

#### Observation: fan-shape sensing region V x {NULL}

### Model - defined same as [paper OO_MOS ICRA2019](https://www.khoury.northeastern.edu/home/lsw/papers/icra2019-mop.pdf) 
#### Transition model - determinstic

#### Observation
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
    -   (by default) uniformly distributed as a historgram
- prior 
    - *line 62, 80* check `class MosOOPOMDP(pomdp_py.OOPOMDP)` in `problem.py`, 

- `belief_update()` in `problem.py`
    - simplifying assumption: object is contained within one pixel (or voxel), so Observation ~ Labelling each cell as OBJ or FREE


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
