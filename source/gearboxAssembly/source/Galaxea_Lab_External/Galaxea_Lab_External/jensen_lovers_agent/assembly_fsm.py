from abc import ABC, abstractmethod

from .agent import PickAndPlaceState

# Interface of state in finite state machine
class State(ABC):
    @abstractmethod
    def update(self, context):
        pass

# State controller
class StateMachine:
    def __init__(self, initial_state, context):
        self.state = initial_state
        self.context = context

    def transition_to(self, next_state):
        self.state = next_state

    def update(self):
        self.state.update(self.context)

class Context:
    NUM_PLANETARY_GEARS = 3
    def __init__(self, sim, agent):
        self.sim = sim
        self.agent = agent
        self.fsm = None            # object of StateMachine
    
    @property
    def is_all_planetary_gear_mounted(self):
        return self.agent.num_mounted_planetary_gears >= self.NUM_PLANETARY_GEARS
        
    @property
    def is_sun_gear_mounted(self):
        return self.agent.is_sun_gear_mounted
    
    @property
    def is_ring_gear_mounted(self):
        return self.agent.is_ring_gear_mounted
    
    @property
    def is_planetary_reducer_mounted(self):
        return self.agent.is_planetary_reducer_mounted
    
# -------------------------------------------------------------------------------------------------------------------------- #
# [FSM Start State] Initialization ----------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
class InitializationState(State):
    def update(self, context):
        # [State Transition] Initialization -> Planetary Gear Mounting
        if context.fsm is not None:
            context.fsm.transition_to(PlanetaryGearMountingState())

# -------------------------------------------------------------------------------------------------------------------------- #
# [FSM Intermediate State] Planetary Gear Mounting ------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
class PlanetaryGearMountingState(State):
    def update(self, context):
        context.agent.pick_and_place()

        if context.agent.pick_and_place_fsm_state == PickAndPlaceState.FINALIZATION:
            if context.is_all_planetary_gear_mounted:
                # [State Transition] Planetary Gear Mounting -> Sun Gear Mounting
                context.fsm.transition_to(SunGearMountingState())

# -------------------------------------------------------------------------------------------------------------------------- #
# [FSM Intermediate State] Sun Gear Mounting ------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
class SunGearMountingState(State):
    def update(self, context):
        context.agent.pick_and_place2(object_name="sun_gear")

        if context.agent.pick_and_place2_fsm_state == PickAndPlaceState.FINALIZATION:
            if context.is_sun_gear_mounted:
                # [State Transition] Sun Gear Mounting -> Ring Gear Mounting
                context.fsm.transition_to(PlanetaryReducerMountingState())

# -------------------------------------------------------------------------------------------------------------------------- #
# [FSM Intermediate State] Ring Gear Mounting ------------------------------------------------------------------------------ #
# -------------------------------------------------------------------------------------------------------------------------- #
# class RingGearMountingState(State):
#     def update(self, context):
#         context.agent.pick_and_place2(object_name="ring_gear")

#         if context.agent.pick_and_place2_fsm_state == PickAndPlaceState.FINALIZATION:
#             if context.is_ring_gear_mounted:
#                 # [State Transition] Ring Gear Mounting -> Planetary Reducer Mounting
#                 context.fsm.transition_to(PlanetaryReducerMountingState())

# -------------------------------------------------------------------------------------------------------------------------- #
# [FSM Intermediate State] Planetary Reducer Mounting ---------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
class PlanetaryReducerMountingState(State):
    def update(self, context):
        context.agent.pick_and_place2(object_name="planetary_reducer")

        if context.agent.pick_and_place2_fsm_state == PickAndPlaceState.FINALIZATION:
            if context.is_planetary_reducer_mounted:
                # [State Transition] Planetary Reducer Mounting -> Finalization
                context.fsm.transition_to(FinalizationState())

# -------------------------------------------------------------------------------------------------------------------------- #
# [FSM End State] Finalization --------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
class FinalizationState(State):
    def update(self, context):
        pass