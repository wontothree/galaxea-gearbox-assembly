from abc import ABC, abstractmethod

# State's life cycle (enter -> update -> exit) interface
class State(ABC):
    def enter(self):
        pass

    @abstractmethod
    def update(self, context):
        pass

    def exit(self):
        pass

# State controller
class StateMachine:
    def __init__(self, initial_state, context):
        self.state = initial_state
        self.context = context
        self.state.enter(self.context)

    def transition_to(self, next_state):
        self.state.exit(self.context)
        self.state = next_state
        self.state.enter(self.context)

    def update(self):
        self.state.update(self.context)

class Context:
    NUM_PLANETARY_GEARS = 3
    def __init__(self, sim, agent):
        self.sim = sim
        self.agent = agent
        self.fsm = None            # object of StateMachine

    # def determine_start_state(self):
    #     if self.is_planetary_reducer_mounted:
    #         return FinalizationState()
    #     elif self.is_ring_gear_mounted:
    #         return PlanetaryReducerMountingState()
    #     elif self.is_sun_gear_mounted:
    #         return RingGearMountingState()
    #     elif self.is_all_planetary_gear_mounted:
    #         return SunGearMountingState()
    #     else:
    #         return PlanetaryGearMountingState()
    
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
    def enter(self):
        print("[FSM Start State] Initialization: enter")
        # context.reset()

    def update(self, context):
        # [State Transition] Initialization -> Planetary Gear Mounting
        if context.fsm is not None:
            context.fsm.transition_to(PlanetaryGearMountingState())

    def exit(self):
        print("[FSM Start State] Initialization: exit")

# -------------------------------------------------------------------------------------------------------------------------- #
# [FSM Intermediate State] Planetary Gear Mounting ------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
class PlanetaryGearMountingState(State):
    def enter(self, context):
        print("[FSM Intermediate State] Planetary Gear Mounting: enter")
        context.agent.reset_pick_and_place()

    def update(self, context):
        context.agent.pick_and_place(object_name="planetary_gear")

        if context.agent.pick_and_place_fsm_state == "FINALIZATION":
            if context.is_all_planetary_gear_mounted:
                # [State Transition] Planetary Gear Mounting -> Sun Gear Mounting
                context.fsm.transition_to(SunGearMountingState())
        
    def exit(self):
        print("[FSM Intermediate State] Planetary Gear Mounting: exit")

# -------------------------------------------------------------------------------------------------------------------------- #
# [FSM Intermediate State] Sun Gear Mounting ------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
class SunGearMountingState(State):
    def enter(self, context):
        print("[FSM Intermediate State] Sun Gear Mouting: enter")
        context.agent.reset_pick_and_twist_insert()
    
    def update(self, context):
        context.agent.pick_and_twist_insert(object_name="sun_gear")

        if context.agent.pick_and_twist_insert_fsm_state == "FINALIZATION":
            if context.is_sun_gear_mounted:
                # [State Transition] Sun Gear Mounting -> Ring Gear Mounting
                context.fsm.transition_to(RingGearMountingState())

    def exit(self):
        print("[FSM Intermediate State] Sun Gear Mounting: exit")

# -------------------------------------------------------------------------------------------------------------------------- #
# [FSM Intermediate State] Ring Gear Mounting ------------------------------------------------------------------------------ #
# -------------------------------------------------------------------------------------------------------------------------- #
class RingGearMountingState(State):
    def enter(self, context):
        print("[FSM Intermediate State] Ring Gear Mouting: enter")
        context.agent.reset_pick_and_twist_insert()
    
    def update(self, context):
        context.agent.pick_and_twist_insert(object_name="ring_gear")

        if context.agent.pick_and_twist_insert_fsm_state == "FINALIZATION":
            if context.is_sun_gear_mounted:
                # [State Transition] Ring Gear Mounting -> Planetary Reducer Mounting
                context.fsm.transition_to(PlanetaryReducerMountingState())

    def exit(self):
        print("[FSM Intermediate State] Ring Gear Mounting: exit")

# -------------------------------------------------------------------------------------------------------------------------- #
# [FSM Intermediate State] Planetary Reducer Mounting ---------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
class PlanetaryReducerMountingState(State):
    def enter(self):
        print("[FSM Intermediate State] Planetary Reducer Mouting: enter")
    
    def update(self, context):
        pass

    def exit(self):
        print("[FSM Intermediate State] Planetary Reducer Mounting: exit")

# -------------------------------------------------------------------------------------------------------------------------- #
# [FSM End State] Finalization --------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------- #
class FinalizationState(State):
    def enter(self):
        print("[FSM Start State] FINALIZATION: enter")

    def update(self, context):
        pass