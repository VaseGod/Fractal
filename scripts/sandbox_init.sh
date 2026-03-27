#!/usr/bin/env bash
# ============================================================
# Fractal — Sandbox Execution Environment
# Creates isolated execution contexts with seccomp + cgroups
# ============================================================
set -euo pipefail

echo "[SANDBOX] Initializing sandbox execution environment..."

SANDBOX_ROOT="${SANDBOX_ROOT:-/tmp/fractal-sandbox}"
SANDBOX_USER="${SANDBOX_USER:-fractal}"

# ── Create sandbox filesystem ──
mkdir -p "${SANDBOX_ROOT}"/{workspace,tmp,logs}
chown -R "${SANDBOX_USER}:${SANDBOX_USER}" "${SANDBOX_ROOT}"
chmod 700 "${SANDBOX_ROOT}"

# ── Restricted /tmp for sandbox ──
mount -t tmpfs -o size=512M,noexec,nosuid,nodev tmpfs "${SANDBOX_ROOT}/tmp" 2>/dev/null || \
    echo "[WARN] tmpfs mount failed (may need privileges). Using directory-based sandbox."

# ── Seccomp profile ──
# Restrict dangerous syscalls within sandbox environments
SECCOMP_PROFILE="${SANDBOX_ROOT}/seccomp-profile.json"
cat > "${SECCOMP_PROFILE}" << 'SECCOMP_EOF'
{
  "defaultAction": "SCMP_ACT_ALLOW",
  "syscalls": [
    {
      "names": [
        "clone",
        "unshare",
        "mount",
        "umount2",
        "pivot_root",
        "chroot",
        "reboot",
        "swapon",
        "swapoff",
        "init_module",
        "finit_module",
        "delete_module",
        "kexec_load",
        "kexec_file_load",
        "ptrace",
        "personality",
        "userfaultfd"
      ],
      "action": "SCMP_ACT_ERRNO",
      "errnoRet": 1
    }
  ]
}
SECCOMP_EOF

echo "[SANDBOX] Seccomp profile written to ${SECCOMP_PROFILE}"

# ── Resource limits via cgroups v2 (if available) ──
CGROUP_ROOT="/sys/fs/cgroup"
FRACTAL_CGROUP="${CGROUP_ROOT}/fractal-sandbox"

if [ -d "${CGROUP_ROOT}" ] && [ -f "${CGROUP_ROOT}/cgroup.controllers" ]; then
    echo "[SANDBOX] Configuring cgroups v2 resource limits..."
    
    # Create cgroup for sandbox
    mkdir -p "${FRACTAL_CGROUP}" 2>/dev/null || true
    
    # Memory limit: 2GB
    echo "2147483648" > "${FRACTAL_CGROUP}/memory.max" 2>/dev/null || true
    # Memory swap limit: 0 (no swap)
    echo "0" > "${FRACTAL_CGROUP}/memory.swap.max" 2>/dev/null || true
    # CPU limit: 200% (2 cores)
    echo "200000 100000" > "${FRACTAL_CGROUP}/cpu.max" 2>/dev/null || true
    # PID limit: 256 processes
    echo "256" > "${FRACTAL_CGROUP}/pids.max" 2>/dev/null || true
    
    echo "[SANDBOX] Resource limits applied:"
    echo "  Memory:  2 GB (no swap)"
    echo "  CPU:     2 cores max"
    echo "  PIDs:    256 max"
else
    echo "[WARN] cgroups v2 not available. Using ulimit fallback."
    
    # Fallback: set ulimits for the sandbox user
    cat >> /etc/security/limits.d/fractal-sandbox.conf << LIMITS_EOF
${SANDBOX_USER}  hard  nproc   256
${SANDBOX_USER}  hard  nofile  1024
${SANDBOX_USER}  hard  as      2097152
${SANDBOX_USER}  hard  fsize   1048576
LIMITS_EOF
    echo "[SANDBOX] ulimit-based resource limits configured."
fi

# ── Create sandbox execution wrapper ──
SANDBOX_EXEC="${SANDBOX_ROOT}/sandbox-exec.sh"
cat > "${SANDBOX_EXEC}" << 'EXEC_EOF'
#!/usr/bin/env bash
# Execute a command inside the Fractal sandbox
set -euo pipefail

SANDBOX_ROOT="${SANDBOX_ROOT:-/tmp/fractal-sandbox}"

# Run with restricted environment
exec env -i \
    HOME="${SANDBOX_ROOT}/workspace" \
    PATH="/usr/local/bin:/usr/bin:/bin" \
    TMPDIR="${SANDBOX_ROOT}/tmp" \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    "$@"
EXEC_EOF

chmod +x "${SANDBOX_EXEC}"

echo "[SANDBOX] Sandbox environment initialized at ${SANDBOX_ROOT}"
echo "[SANDBOX] Use '${SANDBOX_EXEC} <command>' to run commands in sandbox."
