"""TCP forwarder: binds on the Windows-visible interface and forwards to localhost.

Workaround for netsh portproxy not working reliably on WSL2.
Runs on Windows Python — listens on 0.0.0.0:PORT, forwards to WSL_IP:PORT.

Usage (from WSL):
    python.exe scripts/tcp-forward.py 8420 <WSL_IP>

Usage (from PowerShell):
    python scripts/tcp-forward.py 8420 <WSL_IP>
"""

import socket
import sys
import threading


def forward(src, dst):
    try:
        while True:
            data = src.recv(4096)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        src.close()
        dst.close()


def handle(client, target_host, target_port):
    try:
        target = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        target.connect((target_host, target_port))
    except OSError as e:
        print(f"  Cannot reach {target_host}:{target_port}: {e}")
        client.close()
        return

    threading.Thread(target=forward, args=(client, target), daemon=True).start()
    threading.Thread(target=forward, args=(target, client), daemon=True).start()


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <port> <wsl_ip>")
        sys.exit(1)

    port = int(sys.argv[1])
    wsl_ip = sys.argv[2]

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(5)
    print(f"Forwarding 0.0.0.0:{port} -> {wsl_ip}:{port}")

    try:
        while True:
            client, addr = server.accept()
            print(f"  Connection from {addr[0]}:{addr[1]}")
            threading.Thread(target=handle, args=(client, wsl_ip, port), daemon=True).start()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
