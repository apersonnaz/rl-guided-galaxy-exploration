import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from .memoryMonitor import MemoryMonitor


class loggable_operator(object):
    def __call__(self, f):
        def wrapped_f(*args, **kwargs):
            if "logger" in kwargs and kwargs["logger"] != None:
                with ThreadPoolExecutor() as executor:
                    monitor = MemoryMonitor()
                    mem_thread = executor.submit(monitor.measure_usage)
                    startTime = datetime.now()
                    try:
                        f_thread = executor.submit(f, *args, **kwargs)
                        return f_thread.result()
                    finally:
                        monitor.keep_measuring = False
                        log_data = {
                            "module": f.__module__,
                            "operator": f.__name__,
                            "start_time": startTime.timestamp(),
                            "duration": (datetime.now() - startTime).total_seconds(),
                            "peak_memory_usage": mem_thread.result()
                        }
                        kwargs["logger"].append_log(log_data)
            else:
                return f(*args, **kwargs)
        return wrapped_f


class Logger:
    def __init__(self):
        self.log_stack = []
        self.current_log_level = self.log_stack
        self.log_levels_stack = [self.log_stack]

    def start_running_log(self, log_data={}, new_level_name=""):
        self.log_levels_stack.append(self.current_log_level)
        log_data["start_time"] = datetime.now().timestamp()
        self.current_log_level.append(log_data)
        log_data[new_level_name] = []
        self.current_log_level = log_data[new_level_name]

    def append_log(self, log_data={}):
        self.current_log_level.append(log_data)

    def end_running_log(self, additional_log_data={}):
        self.current_log_level = self.log_levels_stack.pop()
        running_log = self.current_log_level[-1]
        running_log.update(additional_log_data)
        running_log["duration"] = (datetime.now(
        ) - datetime.fromtimestamp(running_log["start_time"])).total_seconds()

    def write_log(self, file_name=None):
        file_name = f"./log/log_{datetime.now().strftime('%d-%m-%y-%H:%M:%S')}.json" if file_name == None else file_name
        with open(file_name, "w") as log_file:
            json.dump(self.log_stack, log_file, indent=2)

    def print_log(self):
        print(self.log_stack)

    def concat_logger(self, logger):
        self.log_stack += logger.log_stack

    def log_error(self, error_data):
        while len(self.log_levels_stack) > 1:
            self.end_running_log(additional_log_data=error_data)
