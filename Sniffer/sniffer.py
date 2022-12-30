import argparse
import os, sys
import time

from enum import Enum, auto
from typing import Any

from scapy.layers.inet import IP, TCP
from scapy.sendrecv import AsyncSniffer
from scapy.sessions import DefaultSession
from scapy.utils import wrpcap

#--------------------------------------

PCAP_OUTPUT_PATH = "traffic_collect"
EXPIRED_UPDATE = 40
MAX_PACKETS_IN_FLOW = 10000
MAX_FLOW_DURATION = 120

#--------------------------------------
def generate_session_class():
    return type("NewFlowSession",(FlowSession,),{},)

 # tuple for this flow
def get_packet_flow_key(packet, direction) -> tuple:
    if "TCP" in packet:
        protocol = "TCP"
    elif "UDP" in packet:
        protocol = "UDP"
    else:
        raise Exception("Only TCP protocols are supported.")

    if direction == PacketDirection.FORWARD:
        dest_ip = packet["IP"].dst
        src_ip = packet["IP"].src
        dest_mac = packet["Ether"].dst
        src_mac = packet["Ether"].src
        src_port = packet[protocol].sport
        dest_port = packet[protocol].dport
    else:
        dest_ip = packet["IP"].src
        src_ip = packet["IP"].dst
        dest_mac = packet["Ether"].src
        src_mac = packet["Ether"].dst
        src_port = packet[protocol].dport
        dest_port = packet[protocol].sport

    return dest_ip, src_ip, src_port, dest_port, src_mac, dest_mac

class PacketDirection(Enum):
    FORWARD = auto()
    REVERSE = auto()
    
class Flow:
    def __init__(self, packet: Any, direction: Enum):

        (
            self.dest_ip,
            self.src_ip,
            self.src_port,
            self.dest_port,
            self.src_mac,
            self.dest_mac,
        ) = get_packet_flow_key(packet, direction)
        
        self.packets = []
        self.latest_timestamp = 0
        self.start_timestamp = 0

    def add_packet(self, packet: Any, direction: Enum) -> None:
        self.packets.append((packet, direction))
        self.latest_timestamp = max([packet.time, self.latest_timestamp])
        # First packet of the flow
        if self.start_timestamp == 0:
            self.start_timestamp = packet.time
            self.protocol = packet.proto

    def get_packets(self) -> list:
        return [pkt[0] for pkt in self.packets]

    @property
    def duration(self):
        return self.latest_timestamp - self.start_timestamp
 
class FlowSession(DefaultSession):
    def __init__(self, *args, **kwargs):
        self.flows = {}
        self.flow_id = 0
        self.packets_count = 0
        super(FlowSession, self).__init__(*args, **kwargs)

    def toPacketList(self):
        self.garbage_collect(None)
        print("Parsing " + str(self.flow_id) + " pcap files from sniffer.")
        return super(FlowSession, self).toPacketList()

    def on_packet_received(self, packet):
        count = 0
        direction = PacketDirection.FORWARD
        try:
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get((packet_flow_key, count))
        except Exception:
            return
            
        self.packets_count += 1

        # If there is no forward flow with a count of 0
        if flow is None:
            # There might be one of it in reverse
            direction = PacketDirection.REVERSE
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get((packet_flow_key, count))

        if flow is None:
            # If no flow exists create a new flow
            direction = PacketDirection.FORWARD
            flow = Flow(packet, direction)
            packet_flow_key = get_packet_flow_key(packet, direction)
            self.flows[(packet_flow_key, count)] = flow

        elif (packet.time - flow.latest_timestamp) > EXPIRED_UPDATE:
            # If the packet exists in the flow but the packet is sent
            # after too much of a delay than it is a part of a new flow.
            expired = EXPIRED_UPDATE
            while (packet.time - flow.latest_timestamp) > expired:
                count += 1
                expired += EXPIRED_UPDATE
                flow = self.flows.get((packet_flow_key, count))
                if flow is None:
                    flow = Flow(packet, direction)
                    self.flows[(packet_flow_key, count)] = flow
                    break
        elif "F" in str(packet.flags):
            # If it has FIN flag then early collect flow and continue
            flow.add_packet(packet, direction)
            self.garbage_collect(packet.time)
            return

        flow.add_packet(packet, direction)
        if self.packets_count % MAX_PACKETS_IN_FLOW == 0 or flow.duration > MAX_FLOW_DURATION:
            self.garbage_collect(packet.time)

    def garbage_collect(self, latest_time) -> None:
        global PCAP_OUTPUT_PATH
        keys = list(self.flows.keys())
        for k in keys:
            flow = self.flows.get(k)
            if (latest_time is None or latest_time - flow.latest_timestamp > EXPIRED_UPDATE or flow.duration > 90):
                filename = PCAP_OUTPUT_PATH + "/flow" + str(self.flow_id).zfill(10) + ".pcap"
                packets = flow.get_packets()
                for pkt in packets:
                    wrpcap(filename, pkt, append=True)
                # add lock file
                self.flow_id += 1
                del self.flows[k]


def create_sniffer(input_file, input_interface=None):
    assert (input_file is None) ^ (input_interface is None)
    global PCAP_OUTPUT_PATH
    NewFlowSession = generate_session_class()
    if input_file is not None:
        file_name = os.path.basename(input_file)
        PCAP_OUTPUT_PATH = os.path.splitext(file_name)[0] + time.strftime("-%Y-%m-%d")
        if not os.path.isdir(PCAP_OUTPUT_PATH):
            os.mkdir(PCAP_OUTPUT_PATH);
        return AsyncSniffer(offline=input_file, filter="ip and (tcp or udp)", prn=None, session=NewFlowSession, store=False,)
    else:
        
        if not os.path.isdir(PCAP_OUTPUT_PATH):
            os.mkdir(PCAP_OUTPUT_PATH);
        return AsyncSniffer(iface=input_interface, filter="ip and (tcp or udp)", prn=None, session=NewFlowSession, store=False,)

def main():
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i",
        "--interface",
        action="store",
        dest="input_interface",
    )
    input_group.add_argument(
        "-f",
        "--file",
        action="store",
        dest="input_file",
    )

    args = parser.parse_args()
    sniffer = create_sniffer(args.input_file,args.input_interface,)
    sniffer.start()

    try:
        sniffer.join()
    except KeyboardInterrupt:
        sniffer.stop()
    finally:
        sniffer.join()


if __name__ == "__main__":
    main()
