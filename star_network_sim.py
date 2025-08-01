import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import random
import threading
import time
import queue
from packet_simulator import Packet, PacketGenerator
from sdn_controller import SDNController, QueuePriority
from ml_sdn_controller import MLSDNController
from collections import deque

class StarNetworkSim:
    """Original rule-based SDN controller simulation"""
    def __init__(self):
        """Initialize the star network simulation."""
        # Create an empty graph
        self.G = nx.Graph()
        
        # Add nodes
        self.center_node = "Hub"
        self.peripheral_nodes = ["Node1", "Node2", "Node3", "Node4"]
        
        # Add all nodes to the graph
        self.G.add_node(self.center_node, type='hub')
        for node in self.peripheral_nodes:
            self.G.add_node(node, type='peripheral')
        
        # Add edges (star topology)
        for node in self.peripheral_nodes:
            # Add edge with random latency between 10-50ms
            latency = np.random.uniform(10, 50)
            self.G.add_edge(self.center_node, node, latency=latency)
        
        # Initialize SDN controllers for each node
        self.controllers = {
            self.center_node: SDNController(self.center_node)
        }
        for node in self.peripheral_nodes:
            self.controllers[node] = SDNController(node)
        
        # Start all controllers
        for controller in self.controllers.values():
            controller.start_processing()
        
        # Initialize packet generator
        self.packet_generator = PacketGenerator()
        
        # Fixed positions for the star layout
        self.pos = {
            self.center_node: (0, 0),
            "Node1": (1, 1),
            "Node2": (-1, 1),
            "Node3": (-1, -1),
            "Node4": (1, -1)
        }
        
        # Track active packets
        self.active_packets = []
        self.packet_positions = {}
        
        # Track packet generation
        self.packet_generation_time = time.time()
        self.packet_interval = 0.5  # Generate packet every 0.5 seconds
        
        # Track simulation state
        self.frame_count = 0
        self.total_packets = 0
        
        # Initialize priority counts for each node
        self.priority_counts = {
            node: {priority: 0 for priority in QueuePriority}
            for node in self.G.nodes()
        }
        
        # Initialize metrics tracking
        self.metrics_history = {
            'latency': deque(maxlen=100),
            'throughput': deque(maxlen=100),
            'jitter': deque(maxlen=100),
            'packet_loss': deque(maxlen=100),
            'queue_length': deque(maxlen=100),
            'bandwidth_utilization': deque(maxlen=100)
        }
        
        # Initialize metrics calculation variables
        self.last_packet_time = time.time()
        self.last_latency = None
        self.sent_packets = 0
        self.received_packets = 0
        self.total_latency = 0
        self.total_bandwidth = 0
        self.metrics_update_interval = 0.5
        self.last_metrics_update = time.time()

    def generate_packets(self):
        """Generate packets at regular intervals"""
        current_time = time.time()
        if current_time - self.packet_generation_time >= self.packet_interval:
            # Generate 1-3 packets
            num_packets = random.randint(1, 3)
            packets = self.packet_generator.generate_packets(num_packets)
            
            # Assign random source and destination
            for packet in packets:
                src = random.choice(self.peripheral_nodes)
                dst = random.choice([n for n in self.peripheral_nodes if n != src])
                self.active_packets.append({
                    'packet': packet,
                    'src': src,
                    'dst': dst,
                    'current_node': src,
                    'path': [src, self.center_node, dst],
                    'path_index': 0,
                    'progress': 0.0,
                    'start_time': time.time()  # Record packet creation time
                })
            
            self.total_packets += num_packets
            self.sent_packets += num_packets
            self.packet_generation_time = current_time

    def update_packet_positions(self):
        """Update positions of packets in transit"""
        for packet_info in self.active_packets[:]:
            if packet_info['progress'] >= 1.0:
                # Packet reached next node
                packet_info['path_index'] += 1
                packet_info['progress'] = 0.0
                
                if packet_info['path_index'] >= len(packet_info['path']):
                    # Packet reached destination
                    self.active_packets.remove(packet_info)
                    self.received_packets += 1
                    # Calculate and add latency
                    latency = (time.time() - packet_info['start_time']) * 1000  # Convert to ms
                    self.total_latency += latency
                    continue
                
                # Enqueue packet at current node
                current_node = packet_info['path'][packet_info['path_index']]
                self.controllers[current_node].enqueue_packet(packet_info['packet'])
                
                # Update priority count for the current node
                priority = self.controllers[current_node].determine_priority(packet_info['packet'])
                if priority not in self.priority_counts[current_node]:
                    self.priority_counts[current_node][priority] = 0
                self.priority_counts[current_node][priority] += 1
                
                # Update bandwidth usage
                edge = (packet_info['path'][packet_info['path_index']-1], current_node)
                if edge in self.G.edges():
                    self.total_bandwidth += 1  # Increment bandwidth counter
            
            # Update packet position
            packet_info['progress'] += 0.05  # Move 5% of the way each frame

    def update_metrics(self):
        """Update QoS metrics"""
        current_time = time.time()
        if current_time - self.last_metrics_update >= self.metrics_update_interval:
            # Calculate metrics
            if self.sent_packets > 0:
                # Packet loss rate (ensure it's never negative)
                loss_rate = max(0, ((self.sent_packets - self.received_packets) / self.sent_packets) * 100)
                self.metrics_history['packet_loss'].append(loss_rate)
                
                # Throughput
                throughput = self.received_packets / (current_time - self.last_metrics_update)
                self.metrics_history['throughput'].append(throughput)
                
                # Average latency
                if self.total_latency > 0:
                    avg_latency = self.total_latency / self.received_packets
                    self.metrics_history['latency'].append(avg_latency)
                    
                    # Jitter (difference between consecutive latencies)
                    if self.last_latency is not None:
                        jitter = abs(avg_latency - self.last_latency)
                        self.metrics_history['jitter'].append(jitter)
                    self.last_latency = avg_latency
                
                # Queue length
                total_queue_length = sum(
                    sum(stats.current_size for stats in controller.get_queue_stats().values())
                    for controller in self.controllers.values()
                )
                self.metrics_history['queue_length'].append(total_queue_length)
                
                # Bandwidth utilization (adjusted for more realistic values)
                # Assuming each packet is 1500 bytes (typical MTU) and links are 100Mbps
                if self.total_bandwidth > 0:
                    # Calculate bytes transmitted
                    bytes_transmitted = self.total_bandwidth * 1500  # 1500 bytes per packet
                    bits_transmitted = bytes_transmitted * 8  # Convert to bits
                    
                    # Calculate available bandwidth in the interval
                    time_interval = current_time - self.last_metrics_update
                    available_bits = len(self.G.edges()) * 100 * 1e6 * time_interval  # 100Mbps per link
                    
                    # Calculate utilization percentage
                    utilization = min(100, (bits_transmitted / available_bits) * 100)
                    self.metrics_history['bandwidth_utilization'].append(utilization)
            
            # Reset counters
            self.sent_packets = 0
            self.received_packets = 0
            self.total_latency = 0
            self.total_bandwidth = 0
            self.last_metrics_update = current_time

    def run_simulation(self, frames=None, interval=100, update_queue=None):
        """Run the network simulation"""
        while True:  # Run until interrupted
            # Generate new packets
            self.generate_packets()
            
            # Update packet positions
            self.update_packet_positions()
            
            # Update metrics
            self.update_metrics()
            
            # Send metrics to queue if provided
            if update_queue is not None:
                update_queue.put(('metrics', self.metrics_history))
            
            # Increment frame count and sleep
            self.frame_count += 1
            time.sleep(interval / 1000)  # Convert ms to seconds
        
        # Stop all controllers when simulation ends
        for controller in self.controllers.values():
            controller.stop_processing()

class MLNetworkSim(StarNetworkSim):
    """ML-based SDN controller simulation"""
    def __init__(self):
        """Initialize the ML-based network simulation."""
        super().__init__()
        
        # Replace SDN controllers with ML controllers
        self.controllers = {
            self.center_node: MLSDNController(self.center_node)
        }
        for node in self.peripheral_nodes:
            self.controllers[node] = MLSDNController(node)
        
        # Start all ML controllers
        for controller in self.controllers.values():
            controller.start_processing()

def run_parallel_simulations(frames=None, interval=100):
    """Run both rule-based and ML-based simulations in parallel until interrupted"""
    # Create both simulations
    rule_sim = StarNetworkSim()
    ml_sim = MLNetworkSim()
    
    # Create queues for communication between threads
    rule_queue = queue.Queue()
    ml_queue = queue.Queue()
    
    # Create a single figure with one subplot for network and separate metrics figure
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(12, 8))  # Single figure for network
    ax_network = fig.add_subplot(111)  # Single subplot
    
    # Create metrics figure (keep as is)
    metrics_fig, metrics_axes = plt.subplots(3, 2, figsize=(15, 10))
    metrics_fig.suptitle('Network QoS Metrics (Blue: Rule-based, Red: ML-based)', fontsize=16)
    
    # Initialize metric lines for both simulations
    rule_metric_lines = {}
    ml_metric_lines = {}
    for i, (metric, ax) in enumerate(zip(rule_sim.metrics_history.keys(), metrics_axes.flatten())):
        # Use red for rule-based in latency and jitter, blue for others
        if metric in ['latency', 'jitter']:
            rule_metric_lines[metric] = ax.plot([], [], 'r-')[0]  # Red solid line for rule-based
            ml_metric_lines[metric] = ax.plot([], [], 'b-')[0]  # Blue solid line for ML
        else:
            rule_metric_lines[metric] = ax.plot([], [], 'b-')[0]  # Blue solid line for rule-based
            ml_metric_lines[metric] = ax.plot([], [], 'r-')[0]  # Red solid line for ML
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(True)
        ax.set_xlabel('Time')
        if metric == 'latency':
            ax.set_ylabel('ms')
        elif metric == 'throughput':
            ax.set_ylabel('packets/sec')
        elif metric == 'jitter':
            ax.set_ylabel('ms')
        elif metric == 'packet_loss':
            ax.set_ylabel('%')
        elif metric == 'queue_length':
            ax.set_ylabel('packets')
        elif metric == 'bandwidth_utilization':
            ax.set_ylabel('%')
    
    plt.tight_layout()
    
    # Create threads for each simulation
    def run_rule_sim():
        rule_sim.run_simulation(frames=None, interval=interval, update_queue=rule_queue)
    
    def run_ml_sim():
        ml_sim.run_simulation(frames=None, interval=interval, update_queue=ml_queue)
    
    # Start both simulations in separate threads
    rule_thread = threading.Thread(target=run_rule_sim)
    ml_thread = threading.Thread(target=run_ml_sim)
    
    rule_thread.start()
    ml_thread.start()
    
    # Initialize priority counters with all possible priorities
    def init_priority_counts():
        return {priority: 0 for priority in QueuePriority}
    
    # Map ML model's numeric priorities to QueuePriority
    def map_ml_priority(priority_class):
        """Map ML model's numeric output to QueuePriority enum"""
        # If it's already a QueuePriority enum, return as is
        if isinstance(priority_class, QueuePriority):
            return priority_class
            
        # Convert to float for numeric comparison
        try:
            priority_value = float(priority_class)
            # ML model outputs 0-3 for priority levels
            if priority_value >= 3:
                return QueuePriority.CRITICAL
            elif priority_value >= 2:
                return QueuePriority.HIGH
            elif priority_value >= 1:
                return QueuePriority.MEDIUM
            else:
                return QueuePriority.LOW
        except (ValueError, TypeError):
            return QueuePriority.LOW

    try:
        while True:  # Run until interrupted
            # Clear the network plot
            ax_network.clear()
            ax_network.set_title('Network Simulation (Rule vs ML)')
            
            # Draw edges
            nx.draw_networkx_edges(
                rule_sim.G,
                rule_sim.pos,
                edge_color='black',
                width=2,
                ax=ax_network,
                alpha=0.3  # Make edges semi-transparent
            )
            
            # Draw nodes
            node_colors = ['red' if node == rule_sim.center_node else 'blue' for node in rule_sim.G.nodes()]
            node_sizes = [1000 if node == rule_sim.center_node else 800 for node in rule_sim.G.nodes()]
            nx.draw_networkx_nodes(
                rule_sim.G,
                rule_sim.pos,
                node_color=node_colors,
                node_size=node_sizes,
                ax=ax_network
            )
            
            # Draw rule-based packets (smaller, on the left side)
            for packet_info in rule_sim.active_packets[:]:
                if packet_info['path_index'] >= len(packet_info['path']) - 1:
                    continue
                    
                start_pos = rule_sim.pos[packet_info['path'][packet_info['path_index']]]
                end_pos = rule_sim.pos[packet_info['path'][packet_info['path_index'] + 1]]
                x = start_pos[0] + (end_pos[0] - start_pos[0]) * packet_info['progress'] - 0.1  # Offset left
                y = start_pos[1] + (end_pos[1] - start_pos[1]) * packet_info['progress']
                
                current_node = packet_info['path'][packet_info['path_index']]
                priority = rule_sim.controllers[current_node].determine_priority(packet_info['packet'])
                
                if priority == QueuePriority.CRITICAL:
                    color = '#ff0000'  # bright red
                elif priority == QueuePriority.HIGH:
                    color = '#ff8c00'  # dark orange
                elif priority == QueuePriority.MEDIUM:
                    color = '#ffd700'  # gold
                else:  # LOW
                    color = '#32cd32'  # lime green
                
                ax_network.plot(x, y, 'o', color=color, markersize=8, 
                              markeredgecolor='white', markeredgewidth=1, alpha=0.8)
            
            # Draw ML-based packets (larger, on the right side)
            for packet_info in ml_sim.active_packets[:]:
                if packet_info['path_index'] >= len(packet_info['path']) - 1:
                    continue
                    
                start_pos = ml_sim.pos[packet_info['path'][packet_info['path_index']]]
                end_pos = ml_sim.pos[packet_info['path'][packet_info['path_index'] + 1]]
                x = start_pos[0] + (end_pos[0] - start_pos[0]) * packet_info['progress'] + 0.1  # Offset right
                y = start_pos[1] + (end_pos[1] - start_pos[1]) * packet_info['progress']
                
                current_node = packet_info['path'][packet_info['path_index']]
                priority_value = ml_sim.controllers[current_node].determine_priority(packet_info['packet'])
                priority = map_ml_priority(priority_value)
                
                if priority == QueuePriority.CRITICAL:
                    color = '#ff0000'  # bright red
                elif priority == QueuePriority.HIGH:
                    color = '#ff8c00'  # dark orange
                elif priority == QueuePriority.MEDIUM:
                    color = '#ffd700'  # gold
                else:  # LOW
                    color = '#32cd32'  # lime green
                
                ax_network.plot(x, y, 's', color=color, markersize=10,  # Square markers for ML
                              markeredgecolor='white', markeredgewidth=1)
            
            # Add node labels
            labels = {}
            for node in rule_sim.G.nodes():
                rule_sizes = rule_sim.controllers[node].get_queue_sizes()
                ml_sizes = ml_sim.controllers[node].get_queue_sizes()
                label = f"{node}\nRule | ML\n"
                for priority in QueuePriority:
                    rule_size = rule_sizes.get(priority, 0)
                    ml_size = ml_sizes.get(priority, 0)
                    if rule_size > 0 or ml_size > 0:
                        label += f"{priority.name}: {rule_size} | {ml_size}\n"
                labels[node] = label.strip()
            
            nx.draw_networkx_labels(rule_sim.G, rule_sim.pos, labels=labels, font_size=8, ax=ax_network)
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff0000', 
                          label='CRITICAL', markersize=10, markeredgecolor='white'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff8c00', 
                          label='HIGH', markersize=10, markeredgecolor='white'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffd700', 
                          label='MEDIUM', markersize=10, markeredgecolor='white'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#32cd32', 
                          label='LOW', markersize=10, markeredgecolor='white'),
                plt.Line2D([0], [0], marker='o', color='w', label='Rule-based', markersize=8),
                plt.Line2D([0], [0], marker='s', color='w', label='ML-based', markersize=10)
            ]
            ax_network.legend(handles=legend_elements, loc='upper right', 
                            title='Packet Priorities', fontsize=8)
            
            # Process metrics updates
            while not rule_queue.empty():
                msg_type, data = rule_queue.get_nowait()
                if msg_type == 'metrics':
                    for metric in data:
                        history = list(data[metric])
                        if history:
                            rule_metric_lines[metric].set_data(range(len(history)), history)
            
            while not ml_queue.empty():
                msg_type, data = ml_queue.get_nowait()
                if msg_type == 'metrics':
                    for metric in data:
                        history = list(data[metric])
                        if history:
                            ml_metric_lines[metric].set_data(range(len(history)), history)
            
            # Update axes limits and redraw
            for metric in rule_metric_lines:
                ax = rule_metric_lines[metric].axes
                ax.relim()
                ax.autoscale_view()
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            metrics_fig.canvas.draw()
            metrics_fig.canvas.flush_events()
            plt.pause(0.1)
            
    except KeyboardInterrupt:
        # Clean up
        rule_thread.join()
        ml_thread.join()
        plt.ioff()
        plt.close('all')

if __name__ == "__main__":
    # Run both simulations in parallel until interrupted
    run_parallel_simulations(interval=100)  # Removed frames parameter 