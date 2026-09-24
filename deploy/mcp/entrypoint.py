"""Install a deny-by-default firewall, then permanently drop privileges."""
import ipaddress
import json
import os
import socket
import subprocess
import sys

from psycopg.conninfo import conninfo_to_dict
from policy import DATABASE_ROLE, SEARCH_HOST, public_address


def ipv4_addresses(host):
    return sorted({item[4][0] for item in socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)})


def firewall(destinations):
    # No runtime DNS. Both TLS and libpq connect to the already-resolved IPs.
    for binary in ("iptables", "ip6tables"):
        subprocess.run([binary, "-F", "OUTPUT"], check=True)
        subprocess.run([binary, "-P", "OUTPUT", "DROP"], check=True)
    for address, port in destinations:
        ipaddress.IPv4Address(address)
        subprocess.run(["iptables", "-A", "OUTPUT", "-p", "tcp", "-d", address,
                        "--dport", str(port), "-m", "conntrack", "--ctstate", "NEW,ESTABLISHED", "-j", "ACCEPT"], check=True)


mode = sys.argv[1]
if mode == "broker":
    addresses = ipv4_addresses(SEARCH_HOST)
    if not addresses or not all(public_address(a) for a in addresses):
        raise SystemExit("Search DNS must resolve exclusively to public IPv4 addresses")
    firewall([(address, 443) for address in addresses])
    os.environ["MCP_SEARCH_ADDRESSES"] = ",".join(addresses)
    os.chown("/run/search", 10001, 10001)
    program = "broker.py"
elif mode == "postgres":
    with open("/run/secrets/database.json") as stream:
        secret = json.load(stream)
    connection = conninfo_to_dict(secret["dsn"])
    if connection.get("user") != DATABASE_ROLE or not connection.get("host"):
        raise SystemExit("A dedicated MCP database role and explicit host are required")
    addresses = ipv4_addresses(connection["host"])
    if not addresses:
        raise SystemExit("Database DNS resolution failed")
    port = int(connection.get("port", "5432"))
    firewall([(address, port) for address in addresses])
    os.environ["MCP_DB_ADDRESS"] = addresses[0]
    os.environ["MCP_DATABASE_CONFIG"] = json.dumps(secret)
    program = "server.py"
else:
    raise SystemExit("Unsupported firewall mode")

# No MCP/network-serving process ever runs as root or keeps NET_ADMIN.
os.execvp("setpriv", ["setpriv", "--reuid=10001", "--regid=10001", "--clear-groups",
                      "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all",
                      "--no-new-privs", "python", "-u", "/app/" + program, mode])
