import os

# Save robot_state.txt inside the project folder (relative path)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(PROJECT_ROOT, "robot_state.txt")

def set_state(state):
    """
    state: 'idle' | 'listening' | 'thinking' | 'speaking'
    """
    try:
        with open(STATE_FILE, "w") as f:
            f.write(state)
    except IOError as e:
        print("[STATE] Failed to write state:", e)
