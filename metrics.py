import os
import time
import json
from datetime import datetime

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
            self.ttft = (self.first_token_time - self.start_time) * 1000  # Convert to milliseconds
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
                    topt = (current_time - (self.first_token_time if self.token_count == 1 
                                          else (self.first_token_time + sum(t/1000 for t in self.topt_list)))) * 1000  # Convert to milliseconds
                else:
                    if self.ttft is not None:
                        topt = (current_time - (self.start_time + self.ttft/1000)) * 1000  # Convert to milliseconds
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
        """Average Time Of Per Token in milliseconds (excluding first token)"""
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
            #'topt_list': self.topt_list,
            'total_time': self.total_time,
            'token_count': self.token_count,
            'avg_throughput': self.avg_throughput
        }
    
    def print_metrics(self):
        """Print metrics in a formatted way"""
        metrics = self.get_metrics()
        print("\n=== Inference Performance Metrics ===")
        if metrics['ttft'] is not None:
            print(f"Time To First Token (TTFT): {metrics['ttft']:.2f}ms")
        if metrics['avg_topt'] is not None:
            print(f"Average Time Per Token (TOPT): {metrics['avg_topt']:.2f}ms")
        print(f"Total Generation Time: {metrics['total_time']:.4f}s")
        print(f"Total Tokens Generated: {metrics['token_count']}")
        print(f"Average Throughput: {metrics['avg_throughput']:.2f} tokens/s")
        print("=====================================")

    def save_metrics_to_file(self, save_dir: str = "./outputs/Qwen3-32B"):
        metrics = self.get_metrics()
        metrics["batch_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S.%f")

        filename = "gsm8k_100mbit_client_layers_2.json"
        save_path = os.path.join(save_dir, filename)
        
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            if os.path.exists(save_path):
                with open(save_path, "r", encoding="utf-8") as f:
                    try:
                        metrics_list = json.load(f)
                        if not isinstance(metrics_list, list):
                            metrics_list = []
                    except json.JSONDecodeError:
                        metrics_list = []
            else:
                metrics_list = []

            metrics_list.append(metrics)
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(metrics_list, f, ensure_ascii=False, indent=4)
            
            print(f"\nThe performance indicators have been appended to the file: {save_path}")

        except Exception as e:
            print(f"\nThe performance indicators save to the file failed: {str(e)}")
