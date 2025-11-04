import os, sys, argparse
from scapy.all import *
from scapy.utils import PcapWriter

extensions = {'.pcap', '.pcapng'}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensation', required=True)
    parser.add_argument('--application', required=True)
    parser.add_argument('--new_src_ip', required=True)
    parser.add_argument('--new_dst_ip', required=True)
    parser.add_argument('--new_src_mac', required=True)
    parser.add_argument('--new_dst_mac', required=True)
    parser.add_argument('--new_src_port', type=int, required=True)
    parser.add_argument('--new_dst_port', type=int, required=True)
    return parser.parse_args()

def get_pcap_files(directory):
    """Recursively find all pcap files in subdirectories."""
    pcap_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1] in extensions:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, directory)
                pcap_files.append((full_path, rel_path))
    return pcap_files

def process_pcap(input_path, output_path, new_src_ip, new_dst_ip, new_src_mac, new_dst_mac, new_src_port, new_dst_port):
    """Modify packets and save to output file."""
    try:
        packets = rdpcap(input_path)
        new_cap = PcapWriter(output_path, append=False)

        counter_per_file = 0
        counter_error = 0
        pcap_file_length = len(packets)

        for pkt in packets:
            try:
                eth_layer = Ether(src=new_src_mac, dst=new_dst_mac)
                ip_layer = IP(src=new_src_ip, dst=new_dst_ip)
                udp_layer = UDP(sport=new_src_port, dport=new_dst_port)
                new_pkt = eth_layer / ip_layer / udp_layer / Raw(load=pkt[UDP].payload)
                new_cap.write(new_pkt)
                counter_per_file += 1
            except Exception:
                counter_error += 1
                pass

            sys.stdout.write(f"{input_path}: {counter_per_file}/{pcap_file_length}\r")

        sys.stdout.write(f"\nProcessed: {input_path}, Errors: {counter_error}\n")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    args = get_args()

    input_directory = './pcaps/' + args.sensation + '/' + args.application
    output_directory = './pcaps_output/' + args.sensation + '/' + args.application

    pcap_files = get_pcap_files(input_directory)

    for input_path, rel_path in pcap_files:
        output_path = os.path.join(output_directory, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        process_pcap(
            input_path,
            output_path,
            args.new_src_ip,
            args.new_dst_ip,
            args.new_src_mac,
            args.new_dst_mac,
            args.new_src_port,
            args.new_dst_port
        )

    print("\n[DONE] Processing complete.")

