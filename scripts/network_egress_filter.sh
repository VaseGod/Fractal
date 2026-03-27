#!/usr/bin/env bash
# ============================================================
# Fractal — Network Egress Filtering
# Strict iptables rules: whitelist-only outbound traffic
# Must run as root inside the container
# ============================================================
set -euo pipefail

echo "[SECURITY] Configuring network egress filtering..."

# ── Whitelisted Domains / IPs ──
# Only these endpoints are permitted for outbound connections
ALLOWED_DOMAINS=(
    # Package registries (for initial setup only)
    "pypi.org"
    "files.pythonhosted.org"
    "registry.npmjs.org"
    # LangSmith
    "api.smith.langchain.com"
    # Browserbase
    "api.browserbase.com"
    "connect.browserbase.com"
    # Trigger.dev
    "api.trigger.dev"
    # ARC-AGI-3
    "api.arc-agi.org"
    # ChromaDB (internal — resolved via Docker DNS)
    "chromadb"
    # DNS
    "1.1.1.1"
    "8.8.8.8"
)

# ── Flush existing rules ──
iptables -F OUTPUT 2>/dev/null || true
iptables -F FRACTAL_EGRESS 2>/dev/null || true
iptables -X FRACTAL_EGRESS 2>/dev/null || true

# ── Create custom chain ──
iptables -N FRACTAL_EGRESS

# ── Allow loopback ──
iptables -A FRACTAL_EGRESS -o lo -j ACCEPT

# ── Allow established/related connections ──
iptables -A FRACTAL_EGRESS -m state --state ESTABLISHED,RELATED -j ACCEPT

# ── Allow DNS resolution (UDP 53) to specified DNS servers ──
iptables -A FRACTAL_EGRESS -p udp --dport 53 -d 1.1.1.1 -j ACCEPT
iptables -A FRACTAL_EGRESS -p udp --dport 53 -d 8.8.8.8 -j ACCEPT

# ── Allow internal Docker network traffic ──
# 172.28.0.0/16 is the fractal-internal network
iptables -A FRACTAL_EGRESS -d 172.28.0.0/16 -j ACCEPT

# ── Resolve and whitelist each domain ──
for domain in "${ALLOWED_DOMAINS[@]}"; do
    # Skip non-DNS entries
    if [[ "${domain}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        continue
    fi
    if [[ "${domain}" == "chromadb" ]]; then
        continue
    fi
    
    echo "[SECURITY] Resolving and whitelisting: ${domain}"
    
    # Resolve domain to IPs
    IPS=$(getent hosts "${domain}" 2>/dev/null | awk '{print $1}' || true)
    
    if [ -z "${IPS}" ]; then
        echo "[WARN] Could not resolve ${domain} — skipping"
        continue
    fi
    
    for ip in ${IPS}; do
        iptables -A FRACTAL_EGRESS -d "${ip}" -p tcp --dport 443 -j ACCEPT
        iptables -A FRACTAL_EGRESS -d "${ip}" -p tcp --dport 80 -j ACCEPT
    done
done

# ── Log and drop everything else ──
iptables -A FRACTAL_EGRESS -j LOG \
    --log-prefix "[FRACTAL-EGRESS-DENIED] " \
    --log-level 4 \
    --log-uid
iptables -A FRACTAL_EGRESS -j DROP

# ── Attach custom chain to OUTPUT ──
iptables -A OUTPUT -j FRACTAL_EGRESS

echo "[SECURITY] Egress filtering configured. Only whitelisted domains are permitted."
echo "[SECURITY] Denied connections will be logged with prefix [FRACTAL-EGRESS-DENIED]"

# ── Print active rules for verification ──
echo ""
echo "── Active Egress Rules ──"
iptables -L FRACTAL_EGRESS -n -v --line-numbers
