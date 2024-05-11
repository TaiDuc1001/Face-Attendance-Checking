import time
def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        try:
            message = f"Executed for {func.__name__} with {model_name}: {(end_time-start_time):.4f} seconds."
        except:
            message = f"Executed for {func.__name__}: {(end_time-start_time):.4f} seconds."
        print(message)
        return result
    return wrapper