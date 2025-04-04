class TransitionError(Exception):
    """Raised when operation attempts a state transition that's not allowed
    
    Attributes:
        previous   -- state at beginning of transition
        next_state -- attempt new state
        message    -- explanation of why the specific transition is not allowed
    """

    def __init__(self, previous, next_state, message) -> None:
        self.previous = previous
        self.next = next_state
        self.message = message
