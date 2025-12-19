# POMDP - PY note
reference https://h2r.github.io/pomdp-py/html/examples

## Policy Model
- “prior over actions” part means: instead of sampling uniformly over all actions, the policy model can encode preferences like P(a | state OR history)
    - Many Monte-Carlo planners (POUCT/POMCP) do rollouts/simulations. This [Preference-based action-prior](https://h2r.github.io/pomdp-py/html/examples.action_prior.html) would help 
- instead of hand-coding “uniform random”, you might train a model (from data or RL) that outputs action probabilities given belief/state/history
- for Large (or continuous/parameterized) ACTION space, the distribution is crucial to sample reasonable *range*

## Planner a.k.a Solving process
[Steps](https://h2r.github.io/pomdp-py/html/examples.tiger.html)
1. Create the planner (check `main()`)
2. Agents plans an action
3. Environment state transition according to T(). Reward is returned as a result of transition
4. Agent receives an observation
5. Agent updates history and belief
    - Bellief update also depends on the solver

## Selecting solver
TODO: Have to read about the solver first to understand these parameters
- depth ~ level in the POMCP search tree—how many action/observation steps from the current belief (root)
- num_sims
- rollout_policy


---
# Python-related Question
- what is hashable class and __eq__? TLDR: allow to considered same-name-but-different-obj-instances, __hash__ extend this to be used in sets and dict

- `super()` call parent's function.
    - Base: The class you inherit from (also called “superclass”). In class Child(Base):, Base is the base
    - Parent: The immediate base class of a subclass