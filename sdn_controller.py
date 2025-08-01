from typing import List, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import queue
import threading
import time
from packet_simulator import Packet, DSCP, Protocol, PriorityTag

class QueuePriority(Enum):
    """Queue priorities for packet processing"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

    def __le__(self, other):
        return self.value <= other.value

    def __ge__(self, other):
        return self.value >= other.value

@dataclass
class QueueStats:
    """Statistics for each priority queue"""
    total_packets: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    current_size: int = 0

class SDNController:
    """
    Rule-based SDN controller that processes packets based on header analysis
    and maintains priority queues for packet handling.
    """
    
    def __init__(self, switch_id: str):
        """
        Initialize the SDN controller for a specific switch
        
        Args:
            switch_id: Identifier for the switch this controller manages
        """
        self.switch_id = switch_id
        self.queues = {
            QueuePriority.LOW: queue.Queue(),
            QueuePriority.MEDIUM: queue.Queue(),
            QueuePriority.HIGH: queue.Queue(),
            QueuePriority.CRITICAL: queue.Queue()
        }
        self.queue_stats = {
            priority: QueueStats() for priority in QueuePriority
        }
        self.processing = False
        self.processing_thread = None
        
        # Define DSCP priority mappings
        self.dscp_priority = {
            DSCP.CS0: QueuePriority.LOW,      # Best Effort
            DSCP.CS1: QueuePriority.LOW,      # Priority
            DSCP.AF11: QueuePriority.LOW,     # AF11
            DSCP.AF12: QueuePriority.LOW,     # AF12
            DSCP.AF13: QueuePriority.LOW,     # AF13
            DSCP.CS2: QueuePriority.MEDIUM,   # Immediate
            DSCP.AF21: QueuePriority.MEDIUM,  # AF21
            DSCP.AF22: QueuePriority.MEDIUM,  # AF22
            DSCP.AF23: QueuePriority.MEDIUM,  # AF23
            DSCP.CS3: QueuePriority.MEDIUM,   # Flash
            DSCP.AF31: QueuePriority.HIGH,    # AF31
            DSCP.AF32: QueuePriority.HIGH,    # AF32
            DSCP.AF33: QueuePriority.HIGH,    # AF33
            DSCP.CS4: QueuePriority.HIGH,     # Flash Override
            DSCP.AF41: QueuePriority.HIGH,    # AF41
            DSCP.AF42: QueuePriority.HIGH,    # AF42
            DSCP.AF43: QueuePriority.HIGH,    # AF43
            DSCP.CS5: QueuePriority.CRITICAL, # Critical
            DSCP.EF: QueuePriority.CRITICAL,  # Expedited Forwarding
            DSCP.CS6: QueuePriority.CRITICAL, # Internetwork Control
            DSCP.CS7: QueuePriority.CRITICAL  # Network Control
        }
        
        # Define protocol priority mappings
        self.protocol_priority = {
            Protocol.ICMP: QueuePriority.HIGH,    # Network control
            Protocol.IGMP: QueuePriority.HIGH,    # Multicast control
            Protocol.TCP: QueuePriority.MEDIUM,   # Reliable transport
            Protocol.UDP: QueuePriority.LOW       # Best effort
        }
        
        # Define well-known port priorities
        self.port_priority = {
            # Control and management ports
            22: QueuePriority.HIGH,    # SSH
            23: QueuePriority.HIGH,    # Telnet
            161: QueuePriority.HIGH,   # SNMP
            162: QueuePriority.HIGH,   # SNMP Trap
            179: QueuePriority.HIGH,   # BGP
            520: QueuePriority.HIGH,   # RIP
            53: QueuePriority.HIGH,    # DNS
            # Common service ports
            80: QueuePriority.MEDIUM,  # HTTP
            443: QueuePriority.MEDIUM, # HTTPS
            25: QueuePriority.MEDIUM,  # SMTP
            110: QueuePriority.MEDIUM, # POP3
            143: QueuePriority.MEDIUM, # IMAP
            # Other ports
            5060: QueuePriority.MEDIUM, # SIP
            5061: QueuePriority.MEDIUM, # SIPS
            123: QueuePriority.MEDIUM,  # NTP
        }
    
    def determine_priority(self, packet: Packet) -> QueuePriority:
        """
        Determine the priority of a packet based on header analysis
        
        Args:
            packet: The packet to analyze
            
        Returns:
            QueuePriority: The determined priority level
        """
        # Start with DSCP-based priority
        priority = self.dscp_priority[packet.dscp]
        
        # Adjust based on protocol
        protocol_priority = self.protocol_priority[packet.protocol]
        if protocol_priority.value > priority.value:
            priority = protocol_priority
        
        # Check if source or destination port is in priority list
        if packet.sport in self.port_priority:
            port_priority = self.port_priority[packet.sport]
            if port_priority.value > priority.value:
                priority = port_priority
        
        if packet.dport in self.port_priority:
            port_priority = self.port_priority[packet.dport]
            if port_priority.value > priority.value:
                priority = port_priority
        
        # Special case: ICMP Echo Request/Reply (ping)
        if (packet.protocol == Protocol.ICMP and 
            (packet.sport == 0 or packet.dport == 0)):
            priority = QueuePriority.MEDIUM
        
        return priority
    
    def enqueue_packet(self, packet: Packet) -> None:
        """
        Add a packet to the appropriate priority queue
        
        Args:
            packet: The packet to enqueue
        """
        priority = self.determine_priority(packet)
        self.queues[priority].put(packet)
        self.queue_stats[priority].current_size += 1
        self.queue_stats[priority].total_packets += 1
    
    def process_packet(self, packet: Packet) -> None:
        """
        Process a single packet (simulated)
        
        Args:
            packet: The packet to process
        """
        start_time = time.time()
        
        # Simulate packet processing time based on priority
        if packet.priority_tag == PriorityTag.CRITICAL:
            time.sleep(0.001)  # 1ms processing time
        elif packet.priority_tag == PriorityTag.HIGH:
            time.sleep(0.002)  # 2ms processing time
        elif packet.priority_tag == PriorityTag.MEDIUM:
            time.sleep(0.005)  # 5ms processing time
        else:
            time.sleep(0.01)   # 10ms processing time
        
        processing_time = time.time() - start_time
        
        # Update statistics
        priority = self.determine_priority(packet)
        stats = self.queue_stats[priority]
        stats.total_processing_time += processing_time
        stats.avg_processing_time = (stats.total_processing_time / 
                                   stats.total_packets)
        stats.current_size -= 1
    
    def _process_queues(self) -> None:
        """Process packets from all queues in priority order"""
        while self.processing:
            # Process from highest to lowest priority
            for priority in sorted(QueuePriority, reverse=True):
                try:
                    # Try to get a packet from the current priority queue
                    packet = self.queues[priority].get_nowait()
                    self.process_packet(packet)
                except queue.Empty:
                    continue
            
            # Small delay to prevent CPU hogging
            time.sleep(0.001)
    
    def start_processing(self) -> None:
        """Start the packet processing thread"""
        if not self.processing:
            self.processing = True
            self.processing_thread = threading.Thread(target=self._process_queues)
            self.processing_thread.daemon = True
            self.processing_thread.start()
    
    def stop_processing(self) -> None:
        """Stop the packet processing thread"""
        self.processing = False
        if self.processing_thread:
            self.processing_thread.join()
    
    def get_queue_stats(self) -> Dict[QueuePriority, QueueStats]:
        """Get current statistics for all queues"""
        return self.queue_stats.copy()
    
    def get_queue_sizes(self) -> Dict[QueuePriority, int]:
        """Get current size of all queues"""
        return {priority: stats.current_size 
                for priority, stats in self.queue_stats.items()}

# Example usage
if __name__ == "__main__":
    from packet_simulator import PacketGenerator
    
    # Create a controller for a switch
    controller = SDNController("Switch1")
    
    # Create packet generator
    generator = PacketGenerator()
    
    # Generate some test packets
    test_packets = generator.generate_packets(10)
    
    # Start processing
    controller.start_processing()
    
    # Enqueue packets
    for packet in test_packets:
        controller.enqueue_packet(packet)
        print(f"Enqueued packet: {packet}")
    
    # Wait for processing
    time.sleep(2)
    
    # Print queue statistics
    print("\nQueue Statistics:")
    for priority, stats in controller.get_queue_stats().items():
        print(f"{priority.name}:")
        print(f"  Total packets: {stats.total_packets}")
        print(f"  Average processing time: {stats.avg_processing_time:.6f}s")
        print(f"  Current queue size: {stats.current_size}")
    
    # Stop processing
    controller.stop_processing() 