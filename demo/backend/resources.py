"""Sample process resources during a timed model forward."""
import threading
import time


class ResourceMonitor:
    interval_seconds = 0.01

    def __enter__(self):
        try:
            import psutil
            self.process = psutil.Process()
        except ImportError:
            self.process = None
        self.baseline = self.process.memory_info().rss if self.process else None
        self.peak = self.baseline
        self.stop_event = threading.Event()
        self.thread = None
        if self.process:
            self.thread = threading.Thread(target=self._sample, daemon=True)
            self.thread.start()
        self.cpu_start = time.process_time()
        return self

    def _sample(self):
        while not self.stop_event.wait(self.interval_seconds):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *args):
        self.cpu_seconds = max(0, time.process_time() - self.cpu_start)
        self.stop_event.set()
        if self.thread:
            self.thread.join()
            self.peak = max(self.peak, self.process.memory_info().rss)

    def metrics(self, latency_ms):
        return {"cpu_seconds": self.cpu_seconds,
                "cpu_cores": self.cpu_seconds / (latency_ms / 1000) if latency_ms > 0 else None,
                "rss_peak_mb": self.peak / 1024**2 if self.peak is not None else None,
                "rss_delta_mb": max(0, self.peak - self.baseline) / 1024**2
                if self.peak is not None else None,
                "rss_sample_interval_ms": self.interval_seconds * 1000}
