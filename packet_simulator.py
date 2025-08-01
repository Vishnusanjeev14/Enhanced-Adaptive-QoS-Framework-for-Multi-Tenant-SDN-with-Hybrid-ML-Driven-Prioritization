import random
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass
import ipaddress

class PriorityTag(Enum):
    """Priority levels for packets"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class EthType(Enum):
    """Ethernet types"""
    IPV4 = 0x0800
    IPV6 = 0x86DD
    ARP = 0x0806
    VLAN = 0x8100

class Protocol(Enum):
    """Network protocols"""
    TCP = 6
    UDP = 17
    ICMP = 1
    IGMP = 2

class DSCP(Enum):
    """Differentiated Services Code Point values"""
    CS0 = 0    # Best Effort
    CS1 = 8    # Priority
    AF11 = 10  # Assured Forwarding 11
    AF12 = 12  # Assured Forwarding 12
    AF13 = 14  # Assured Forwarding 13
    CS2 = 16   # Immediate
    AF21 = 18  # Assured Forwarding 21
    AF22 = 20  # Assured Forwarding 22
    AF23 = 22  # Assured Forwarding 23
    CS3 = 24   # Flash
    AF31 = 26  # Assured Forwarding 31
    AF32 = 28  # Assured Forwarding 32
    AF33 = 30  # Assured Forwarding 33
    CS4 = 32   # Flash Override
    AF41 = 34  # Assured Forwarding 41
    AF42 = 36  # Assured Forwarding 42
    AF43 = 38  # Assured Forwarding 43
    CS5 = 40   # Critical
    EF = 46    # Expedited Forwarding
    CS6 = 48   # Internetwork Control
    CS7 = 56   # Network Control

@dataclass
class Packet:
    """Packet class representing network packets with various headers"""
    priority_tag: PriorityTag
    eth_type: EthType
    dscp: DSCP
    src: str
    dst: str
    protocol: Protocol
    sport: int
    dport: int
    
    def __post_init__(self):
        """Validate packet fields after initialization"""
        self._validate_ports()
        self._validate_ip_addresses()
    
    def _validate_ports(self) -> None:
        """Validate source and destination ports"""
        if not (0 <= self.sport <= 65535):
            raise ValueError(f"Invalid source port: {self.sport}")
        if not (0 <= self.dport <= 65535):
            raise ValueError(f"Invalid destination port: {self.dport}")
    
    def _validate_ip_addresses(self) -> None:
        """Validate source and destination IP addresses"""
        try:
            ipaddress.ip_address(self.src)
            ipaddress.ip_address(self.dst)
        except ValueError as e:
            raise ValueError(f"Invalid IP address: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert packet to dictionary format"""
        return {
            'priority_tag': self.priority_tag.name,
            'eth_type': self.eth_type.name,
            'dscp': self.dscp.name,
            'src': self.src,
            'dst': self.dst,
            'protocol': self.protocol.name,
            'sport': self.sport,
            'dport': self.dport
        }
    
    def __str__(self) -> str:
        """String representation of the packet"""
        return (f"Packet(src={self.src}:{self.sport}, dst={self.dst}:{self.dport}, "
                f"proto={self.protocol.name}, priority={self.priority_tag.name}, "
                f"eth_type={self.eth_type.name}, dscp={self.dscp.name})")

class PacketGenerator:
    """Generates random network packets with valid configurations"""
    
    def __init__(self, 
                 src_ips: Optional[list[str]] = None,
                 dst_ips: Optional[list[str]] = None,
                 src_ports: Optional[list[int]] = None,
                 dst_ports: Optional[list[int]] = None):
        """
        Initialize packet generator with optional predefined values
        
        Args:
            src_ips: List of source IP addresses
            dst_ips: List of destination IP addresses
            src_ports: List of source ports
            dst_ports: List of destination ports
        """
        self.src_ips = src_ips or ['192.168.1.1', '192.168.1.2', '192.168.1.3']
        self.dst_ips = dst_ips or ['10.0.0.1', '10.0.0.2', '10.0.0.3']
        self.src_ports = src_ports or list(range(1024, 65536))
        self.dst_ports = dst_ports or list(range(1, 1024))
        
        # Validate provided values
        self._validate_ips()
        self._validate_ports()
    
    def _validate_ips(self) -> None:
        """Validate IP addresses"""
        for ip in self.src_ips + self.dst_ips:
            try:
                ipaddress.ip_address(ip)
            except ValueError as e:
                raise ValueError(f"Invalid IP address in configuration: {ip}")
    
    def _validate_ports(self) -> None:
        """Validate port ranges"""
        for port in self.src_ports + self.dst_ports:
            if not (0 <= port <= 65535):
                raise ValueError(f"Invalid port in configuration: {port}")
    
    def generate_packet(self) -> Packet:
        """Generate a random packet with valid configuration"""
        return Packet(
            priority_tag=random.choice(list(PriorityTag)),
            eth_type=random.choice(list(EthType)),
            dscp=random.choice(list(DSCP)),
            src=random.choice(self.src_ips),
            dst=random.choice(self.dst_ips),
            protocol=random.choice(list(Protocol)),
            sport=random.choice(self.src_ports),
            dport=random.choice(self.dst_ports)
        )
    
    def generate_packets(self, count: int) -> list[Packet]:
        """Generate multiple random packets"""
        if count < 1:
            raise ValueError("Count must be at least 1")
        return [self.generate_packet() for _ in range(count)]

# Example usage
if __name__ == "__main__":
    # Create packet generator with default configuration
    generator = PacketGenerator()
    
    # Generate a single packet
    packet = generator.generate_packet()
    print("Single packet:")
    print(packet)
    print("Packet as dictionary:")
    print(packet.to_dict())
    
    # Generate multiple packets
    print("\nMultiple packets:")
    packets = generator.generate_packets(3)
    for p in packets:
        print(p) 