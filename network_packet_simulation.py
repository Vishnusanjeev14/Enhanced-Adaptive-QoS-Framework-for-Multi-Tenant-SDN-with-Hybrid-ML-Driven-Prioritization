import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import random
import threading
import time
from packet_simulator import Packet, PacketGenerator
from sdn_controller import SDNController, QueuePriority

class StarNetworkSim:
    """
    Simulates a star topology network with SDN controllers and packet processing.
    """
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
        
        # Set up the visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Fixed positions for the star layout
        self.pos = {
            self.center_node: (0, 0),
            "Node1": (1, 1),
            "Node2": (-1, 1),
            "Node3": (-1, -1),
            "Node4": (1, -1)
        }
        
        # Initialize node colors and sizes
        self.node_colors = ['red'] + ['blue'] * 4
        self.node_sizes = [1000] + [800] * 4
        
        # Initialize edge colors
        self.edge_colors = ['black'] * 4
        
        # Track active packets
        self.active_packets = []
        self.packet_positions = {}
        
        # Track packet generation
        self.last_packet_time = time.time()
        self.packet_interval = 0.5  # Generate packet every 0.5 seconds
        
        # Track simulation state
        self.frame_count = 0
        self.total_packets = 0
        
        # Initialize priority counts for each node
        self.priority_counts = {
            node: {priority: 0 for priority in QueuePriority}
            for node in self.G.nodes()
        }

    def generate_packets(self):
        """Generate packets at regular intervals"""
        current_time = time.time()
        if current_time - self.last_packet_time >= self.packet_interval:
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
                    'progress': 0.0
                })
            
            self.total_packets += num_packets
            self.last_packet_time = current_time

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
                    continue
                
                # Enqueue packet at current node
                current_node = packet_info['path'][packet_info['path_index']]
                self.controllers[current_node].enqueue_packet(packet_info['packet'])
                
                # Update priority count for the current node
                priority = self.controllers[current_node].determine_priority(packet_info['packet'])
                self.priority_counts[current_node][priority] += 1
            
            # Update packet position
            packet_info['progress'] += 0.05  # Move 5% of the way each frame

    def get_packet_position(self, packet_info):
        """Calculate current position of a packet"""
        if packet_info['path_index'] >= len(packet_info['path']) - 1:
            return None
        
        current_node = packet_info['path'][packet_info['path_index']]
        next_node = packet_info['path'][packet_info['path_index'] + 1]
        
        start_pos = self.pos[current_node]
        end_pos = self.pos[next_node]
        
        return (
            start_pos[0] + (end_pos[0] - start_pos[0]) * packet_info['progress'],
            start_pos[1] + (end_pos[1] - start_pos[1]) * packet_info['progress']
        )

    def update(self, frame):
        """Update function for animation"""
        self.ax.clear()
        
        # Generate new packets
        self.generate_packets()
        
        # Update packet positions
        self.update_packet_positions()
        
        # Draw edges
        nx.draw_networkx_edges(
            self.G,
            self.pos,
            edge_color=self.edge_colors,
            width=2,
            ax=self.ax
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.G,
            self.pos,
            node_color=self.node_colors,
            node_size=self.node_sizes,
            ax=self.ax
        )
        
        # Draw packets
        for packet_info in self.active_packets:
            pos = self.get_packet_position(packet_info)
            if pos:
                # Draw packet as a small circle
                self.ax.plot(pos[0], pos[1], 'o', color='green', markersize=10)
        
        # Add node labels with queue sizes and total counts
        labels = {}
        for node in self.G.nodes():
            queue_sizes = self.controllers[node].get_queue_sizes()
            label = f"{node}\n"
            for priority in QueuePriority:
                current_size = queue_sizes[priority]
                total_count = self.priority_counts[node][priority]
                if current_size > 0 or total_count > 0:
                    label += f"{priority.name}: {current_size} (Total: {total_count})\n"
            labels[node] = label.strip()
        
        nx.draw_networkx_labels(
            self.G,
            self.pos,
            labels=labels,
            font_size=8,
            ax=self.ax
        )
        
        # Add edge labels (latency)
        edge_labels = {(u, v): f"{self.G[u][v]['latency']:.1f}ms" 
                      for u, v in self.G.edges()}
        nx.draw_networkx_edge_labels(
            self.G,
            self.pos,
            edge_labels=edge_labels,
            ax=self.ax
        )
        
        # Update title with statistics
        self.ax.set_title(
            f"Star Network Simulation\n"
            f"Frame: {frame} | Total Packets: {self.total_packets}\n"
            f"Active Packets: {len(self.active_packets)}"
        )
        
        self.frame_count += 1
        return []

    def run_simulation(self, frames=200, interval=100):
        """Run the network simulation animation"""
        self.anim = FuncAnimation(
            self.fig,
            self.update,
            frames=frames,
            interval=interval,
            blit=True
        )
        plt.show()
        
        # Stop all controllers when simulation ends
        for controller in self.controllers.values():
            controller.stop_processing()

if __name__ == "__main__":
    # Create and run the simulation
    sim = StarNetworkSim()
    sim.run_simulation(frames=200, interval=100) 