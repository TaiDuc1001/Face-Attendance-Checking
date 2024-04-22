import time
def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Executed for {func.__name__}: {end_time-start_time} seconds.")
        return result
    return wrapper