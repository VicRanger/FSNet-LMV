import time


class Timer:
    def __init__(self):
        self.time_record = []
        self.is_recording = False
        self.time_tick = 0
        self.time_saved = 0
    
    def start(self):
        if self.is_recording:
            raise RuntimeError("cant start a running timer")
        self.time_tick = time.time()
        self.is_recording = True
    
    def tick(self):
        self.stop()
        self.start()

    def pause(self):
        if not self.is_recording:
            raise RuntimeError("cant pause an idle timer")
        self.time_saved += time.time() - self.time_tick
        self.is_recording = False

    def stop(self):
        if not self.is_recording:
            raise RuntimeError("cant stop an idle timer")
        self.pause()
        self.time_record.append(self.time_saved)
        self.time_saved = 0

        
    def get_last_time(self):
        if len(self.time_record) <= 0:
            raise RuntimeError("no time record")
        return self.time_record[-1]
    
    def get_avg_time(self):
        if len(self.time_record) <= 0:
            raise RuntimeError("no time record")
        ret = 0
        for t in self.time_record:
            ret += t
        return ret / float(len(self.time_record))
