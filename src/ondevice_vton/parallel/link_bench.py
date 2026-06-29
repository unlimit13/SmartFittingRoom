#!/usr/bin/env python3
"""Stage 0 (torch-free): measure raw TCP throughput + RTT over the direct link.

gloo runs collectives over TCP, so raw point-to-point TCP bandwidth/latency is a
good proxy for whether the link can support 2-way tensor parallelism, WITHOUT
installing PyTorch on the peer. Uses only the Python standard library.

Run the server on rank 0 (its eth0 IP is --host), the client on rank 1:

  # rank 0  (192.168.100.1):
  python3 parallel/link_bench.py --role server --host 192.168.100.1
  # rank 1  (192.168.100.2):
  python3 parallel/link_bench.py --role client --host 192.168.100.1

Measures: client->server bulk throughput (MB/s) and small-message ping-pong RTT.
"""
import argparse
import socket
import time

MAGIC = b"OK"


def recv_exact(conn, n, buf):
    got = 0
    while got < n:
        chunk = conn.recv_into(buf, min(len(buf), n - got))
        if chunk == 0:
            raise ConnectionError("peer closed early")
        got += chunk
    return got


def server(host, port, total, msg, lat_size, lat_count):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    print(f"[server] listening on {host}:{port} ...", flush=True)
    conn, addr = s.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[server] peer {addr} connected", flush=True)

    # ---- throughput: read exactly `total` bytes, time first->last ----
    buf = bytearray(msg)
    first = conn.recv_into(buf, len(buf))
    t0 = time.time()
    got = first
    while got < total:
        n = conn.recv_into(buf, min(len(buf), total - got))
        if n == 0:
            raise ConnectionError("peer closed early")
        got += n
    dt = time.time() - t0
    mbps = total / 1e6 / dt
    print(f"\n[throughput] recv {total/1e6:.0f} MB in {dt:.2f}s "
          f"-> {mbps:.1f} MB/s ({mbps*8:.0f} Mbps)", flush=True)
    conn.sendall(MAGIC)

    # ---- latency: echo `lat_count` small messages ----
    lbuf = bytearray(lat_size)
    for _ in range(lat_count):
        recv_exact(conn, lat_size, lbuf)
        conn.sendall(lbuf)
    print("[server] latency echo done", flush=True)
    conn.close()
    s.close()


def client(host, port, total, msg, lat_size, lat_count):
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.connect((host, port))
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[client] connected to {host}:{port}", flush=True)

    # ---- throughput: send exactly `total` bytes ----
    payload = b"\xab" * msg
    sent = 0
    t0 = time.time()
    while sent < total:
        n = min(msg, total - sent)
        conn.sendall(payload[:n])
        sent += n
    ack = conn.recv(len(MAGIC))
    dt = time.time() - t0
    print(f"\n[throughput] sent {total/1e6:.0f} MB, server-ack in {dt:.2f}s "
          f"-> {total/1e6/dt:.1f} MB/s ({total/1e6/dt*8:.0f} Mbps)", flush=True)

    # ---- latency: ping-pong, measure RTT ----
    small = b"\xcd" * lat_size
    lbuf = bytearray(lat_size)
    rtts = []
    for _ in range(lat_count):
        t = time.time()
        conn.sendall(small)
        recv_exact(conn, lat_size, lbuf)
        rtts.append(time.time() - t)
    rtts.sort()
    avg = sum(rtts) / len(rtts)
    p50 = rtts[len(rtts) // 2]
    p99 = rtts[int(len(rtts) * 0.99)]
    print(f"[latency] {lat_size}B ping-pong x{lat_count}: "
          f"avg {avg*1e6:.0f}us  p50 {p50*1e6:.0f}us  p99 {p99*1e6:.0f}us", flush=True)

    # ---- rough TP verdict using measured throughput ----
    mbps = total / 1e6 / dt
    EST_COMM_MB = 350.0   # rough per-forward all-gather volume, 2-way TP @1024x768
    COMPUTE_TP_S = 60.0
    BASE_S = 119.0
    comm_s = EST_COMM_MB / mbps
    tp_step = COMPUTE_TP_S + comm_s
    speedup = BASE_S / tp_step
    print("\n[verdict] 2-way TP per-step estimate (rough, raw-TCP proxy):")
    print(f"  link throughput        : {mbps:.0f} MB/s")
    print(f"  est comm/forward (~{EST_COMM_MB:.0f}MB): {comm_s:.1f} s")
    print(f"  est compute (TP ~half) : {COMPUTE_TP_S:.0f} s")
    print(f"  => TP step ~{tp_step:.0f}s vs single-Pi {BASE_S:.0f}s -> {speedup:.2f}x")
    print("  PASS (>=1.4x): proceed to Stage 1 (install torch on peer)."
          if speedup >= 1.4 else
          "  MARGINAL: tune comm (fp16/overlap) or reconsider before investing.")
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["server", "client"], required=True)
    ap.add_argument("--host", required=True, help="server (rank 0) eth0 IP")
    ap.add_argument("--port", type=int, default=29555)
    ap.add_argument("--mb", type=int, default=200, help="bulk transfer size (MB)")
    ap.add_argument("--msg", type=int, default=1 << 20, help="chunk bytes")
    ap.add_argument("--lat-size", type=int, default=64)
    ap.add_argument("--lat-count", type=int, default=1000)
    args = ap.parse_args()
    total = args.mb * 1024 * 1024
    fn = server if args.role == "server" else client
    fn(args.host, args.port, total, args.msg, args.lat_size, args.lat_count)


if __name__ == "__main__":
    main()
