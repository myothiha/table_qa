import signal
import time

def handler(signum, frame):
    raise TimeoutError("Timed out!")

signal.signal(signal.SIGALRM, handler)
signal.alarm(2)  # Set 2-second timeout

try:
    print("Sleeping for 10 seconds...")
    time.sleep(10)
    print("Done sleeping.")
except TimeoutError as e:
    print("Caught timeout:", e)
finally:
    signal.alarm(0)  # Clear alarm