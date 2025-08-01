import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import queue
import threading
import time
from joblib import load
from packet_simulator import Packet, DSCP, Protocol, PriorityTag, EthType

class QueuePriority(Enum):
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
    total_packets: int = 0
    processing_time: float = 0.0
    avg_processing_time: float = 0.0
    current_size: int = 0

class MLSDNController:
    """
    Hybrid SDN controller that uses both rule-based DSCP analysis and ML-based priority assignment.
    """
    def __init__(self, node_name: str):
        """
        Initialize the ML-based SDN controller.
        
        Args:
            node_name: Name of the node this controller manages
        """
        self.node_name = node_name
        self.queues = {priority: queue.Queue() for priority in QueuePriority}
        self.queue_stats = {priority: QueueStats() for priority in QueuePriority}
        self.processing = False
        self.processing_thread = None
        
        # Load the ML model
        try:
            self.ml_model = load('ml_priority_model.joblib')
            self.use_ml = True
        except Exception:
            self.ml_model = None
            self.use_ml = False
        
        # DSCP priority mapping for critical packets
        self.dscp_priority_map = {
            DSCP.CS7: QueuePriority.CRITICAL,  # Network Control
            DSCP.CS6: QueuePriority.CRITICAL,  # Internetwork Control
            DSCP.EF: QueuePriority.CRITICAL,   # Expedited Forwarding
            DSCP.AF41: QueuePriority.HIGH,     # Assured Forwarding
            DSCP.AF42: QueuePriority.HIGH,
            DSCP.AF43: QueuePriority.HIGH,
            DSCP.AF31: QueuePriority.MEDIUM,
            DSCP.AF32: QueuePriority.MEDIUM,
            DSCP.AF33: QueuePriority.MEDIUM,
            DSCP.AF21: QueuePriority.MEDIUM,
            DSCP.AF22: QueuePriority.MEDIUM,
            DSCP.AF23: QueuePriority.MEDIUM,
            DSCP.AF11: QueuePriority.LOW,
            DSCP.AF12: QueuePriority.LOW,
            DSCP.AF13: QueuePriority.LOW,
            DSCP.CS0: QueuePriority.LOW         # Best Effort
        }
        
        # Protocol priority mapping
        self.protocol_priority_map = {
            Protocol.ICMP: QueuePriority.HIGH,
            Protocol.TCP: QueuePriority.MEDIUM,
            Protocol.UDP: QueuePriority.LOW
        }
        
        # Well-known port priorities
        self.port_priority_map = {
            # Critical services
            22: QueuePriority.CRITICAL,    # SSH
            53: QueuePriority.CRITICAL,    # DNS
            80: QueuePriority.HIGH,        # HTTP
            443: QueuePriority.HIGH,       # HTTPS
            5060: QueuePriority.HIGH,      # SIP
            5061: QueuePriority.HIGH,      # SIPS
            # High priority services
            20: QueuePriority.HIGH,        # FTP Data
            21: QueuePriority.HIGH,        # FTP Control
            25: QueuePriority.HIGH,        # SMTP
            110: QueuePriority.HIGH,       # POP3
            143: QueuePriority.HIGH,       # IMAP
            # Medium priority services
            123: QueuePriority.MEDIUM,     # NTP
            161: QueuePriority.MEDIUM,     # SNMP
            162: QueuePriority.MEDIUM,     # SNMP Trap
            # Default for other ports
            'default': QueuePriority.LOW
        }

    def prepare_ml_features(self, packet: Packet) -> pd.DataFrame:
        """Prepare features for ML model prediction"""
        # Create a dictionary of features with correct names matching training data
        features = {
            'vlan_priority': float(packet.priority_tag.value),
            'eth_type': float(packet.eth_type.value),
            'dscp': float(packet.dscp.value),
            'src_ip': float(int(packet.src.split('.')[-1])),  # Use last octet of IP
            'dst_ip': float(int(packet.dst.split('.')[-1])),
            'protocol': float(packet.protocol.value),
            'src_port': float(packet.sport),
            'dst_port': float(packet.dport)
        }
        
        # Convert to DataFrame with proper feature names
        df = pd.DataFrame([features])
        return df

    def fallback_priority(self, packet: Packet) -> QueuePriority:
        """Rule-based fallback priority determination"""
        # Check protocol priority
        protocol_priority = self.protocol_priority_map.get(packet.protocol, QueuePriority.LOW)
        
        # Check port priority
        port_priority = self.port_priority_map.get(packet.dport, 
                                                 self.port_priority_map.get(packet.sport, 
                                                                          self.port_priority_map['default']))
        
        # Check DSCP priority
        dscp_priority = self.dscp_priority_map.get(packet.dscp, QueuePriority.LOW)
        
        # Return highest priority among all factors
        return max(protocol_priority, port_priority, dscp_priority)

    def determine_priority(self, packet: Packet) -> QueuePriority:
        """Determine packet priority using hybrid approach"""
        # First check if packet is critical based on DSCP
        if packet.dscp in self.dscp_priority_map:
            return self.dscp_priority_map[packet.dscp]
        
        # For non-critical packets, try to use ML model if available
        if self.use_ml and self.ml_model is not None:
            try:
                # Prepare features and get ML prediction
                features_df = self.prepare_ml_features(packet)
                priority_value = float(self.ml_model.predict(features_df)[0])
                
                # Map prediction to QueuePriority
                if priority_value >= 3:
                    return QueuePriority.CRITICAL
                elif priority_value >= 2:
                    return QueuePriority.HIGH
                elif priority_value >= 1:
                    return QueuePriority.MEDIUM
                else:
                    return QueuePriority.LOW
                    
            except Exception:
                return self.fallback_priority(packet)
        
        # If ML is not available or failed, use rule-based priority
        return self.fallback_priority(packet)

    def enqueue_packet(self, packet: Packet) -> None:
        """
        Add a packet to the appropriate priority queue.
        
        Args:
            packet: The packet to enqueue
        """
        priority = self.determine_priority(packet)
        self.queues[priority].put(packet)
        self.queue_stats[priority].current_size += 1
        self.queue_stats[priority].total_packets += 1

    def process_packet(self, priority: QueuePriority) -> None:
        """
        Process a packet from the specified priority queue.
        
        Args:
            priority: The priority queue to process from
        """
        try:
            packet = self.queues[priority].get_nowait()
            start_time = time.time()
            
            # Simulate processing time based on priority
            if priority == QueuePriority.CRITICAL:
                time.sleep(0.001)  # 1ms for critical
            elif priority == QueuePriority.HIGH:
                time.sleep(0.002)  # 2ms for high
            elif priority == QueuePriority.MEDIUM:
                time.sleep(0.003)  # 3ms for medium
            else:
                time.sleep(0.005)  # 5ms for low
            
            processing_time = time.time() - start_time
            stats = self.queue_stats[priority]
            stats.processing_time += processing_time
            stats.avg_processing_time = stats.processing_time / stats.total_packets
            stats.current_size -= 1
            
        except queue.Empty:
            pass

    def _process_queues(self) -> None:
        """Process packets from all queues in priority order."""
        while self.processing:
            # Process in priority order
            for priority in sorted(QueuePriority, reverse=True):
                self.process_packet(priority)
            time.sleep(0.001)  # Small delay to prevent CPU hogging

    def start_processing(self) -> None:
        """Start processing packets from the queues."""
        if not self.processing:
            self.processing = True
            self.processing_thread = threading.Thread(target=self._process_queues)
            self.processing_thread.daemon = True
            self.processing_thread.start()

    def stop_processing(self) -> None:
        """Stop processing packets."""
        self.processing = False
        if self.processing_thread:
            self.processing_thread.join()

    def get_queue_sizes(self) -> Dict[QueuePriority, int]:
        """Get current size of each priority queue."""
        return {priority: self.queue_stats[priority].current_size 
                for priority in QueuePriority}

    def get_queue_stats(self) -> Dict[QueuePriority, QueueStats]:
        """Get statistics for each priority queue."""
        return self.queue_stats.copy() 