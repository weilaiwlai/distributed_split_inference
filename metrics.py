import os
import time

class Metrics:
    """Metrics class for tracking inference performance"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.ttft = None  # Time To First Token
        self.topt_list = []  # Time Of Per Token list
        self.total_time = 0
        self.token_count = 0
        self.start_time = None
        self.first_token_time = None
        self.is_generating = False
    
    def start_generation(self):
        """Start timing generation process"""
        self.reset()
        self.start_time = time.time()
        self.is_generating = True
    
    def record_first_token(self):
        """Record time for first token"""
        if self.is_generating and self.start_time is not None and self.ttft is None:
            self.first_token_time = time.time()
            self.ttft = self.first_token_time - self.start_time
            self.token_count = 1
    
    def record_next_token(self):
        """Record time for subsequent tokens"""
        if self.is_generating and self.start_time is not None:
            current_time = time.time()
            if self.token_count == 0:
                # First token
                self.record_first_token()
            else:
                # Calculate time for this token
                if len(self.topt_list) > 0:
                    # Use previous token time to calculate TOPT
                    topt = current_time - (self.first_token_time if self.token_count == 1 
                                          else (self.first_token_time + sum(self.topt_list)))
                else:
                    if self.ttft is not None:
                        topt = current_time - (self.start_time + self.ttft)
                    else:
                        # If first token not recorded yet, treat as first
                        self.record_first_token()
                        return
                self.topt_list.append(topt)
                self.token_count += 1
    
    def end_generation(self):
        """End timing generation process"""
        if self.is_generating and self.start_time is not None:
            self.total_time = time.time() - self.start_time
            self.is_generating = False
    
    @property
    def avg_topt(self):
        """Average Time Of Per Token (excluding first token)"""
        if len(self.topt_list) > 0:
            return sum(self.topt_list) / len(self.topt_list)
        return None
    
    @property
    def avg_throughput(self):
        """Average tokens per second"""
        if self.total_time > 0:
            return self.token_count / self.total_time
        return 0
    
    def get_metrics(self):
        """Get all metrics as a dictionary"""
        return {
            'ttft': self.ttft,
            'avg_topt': self.avg_topt,
            'topt_list': self.topt_list,
            'total_time': self.total_time,
            'token_count': self.token_count,
            'avg_throughput': self.avg_throughput
        }
    
    def print_metrics(self):
        """Print metrics in a formatted way"""
        metrics = self.get_metrics()
        print("\n=== Inference Performance Metrics ===")
        if metrics['ttft'] is not None:
            print(f"Time To First Token (TTFT): {metrics['ttft']:.4f}s")
        if metrics['avg_topt'] is not None:
            print(f"Average Time Per Token (TOPT): {metrics['avg_topt']:.4f}s")
        print(f"Total Generation Time: {metrics['total_time']:.4f}s")
        print(f"Total Tokens Generated: {metrics['token_count']}")
        print(f"Average Throughput: {metrics['avg_throughput']:.2f} tokens/s")
        print("=====================================")
