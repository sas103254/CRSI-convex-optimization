"""
Prints the hardware and software details needed for the
"Implementation and Reproducibility" subsection (Sec. VI-B) of the paper,
and saves them to ../results/environment.json.

Run it on the SAME machine that produced the runtime numbers:

    python3 env_info.py

Works on Windows, macOS, and Linux. Uses only the standard library
(plus psutil if it happens to be installed, for RAM and clock speed).
"""
import json
import os
import platform
import subprocess
import sys
from importlib import metadata


def _run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ''


def cpu_model():
    system = platform.system()
    if system == 'Windows':
        out = _run('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name"')
        return out or platform.processor()
    if system == 'Darwin':
        return _run('sysctl -n machdep.cpu.brand_string') or platform.processor()
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.lower().startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def physical_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except ImportError:
        pass
    system = platform.system()
    if system == 'Windows':
        out = _run('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | '
                   'Measure-Object -Property NumberOfCores -Sum).Sum"')
    elif system == 'Darwin':
        out = _run('sysctl -n hw.physicalcpu')
    else:
        out = _run("lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l")
    return int(out) if out.isdigit() else None


def ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except ImportError:
        pass
    system = platform.system()
    if system == 'Windows':
        out = _run('powershell -NoProfile -Command "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"')
    elif system == 'Darwin':
        out = _run('sysctl -n hw.memsize')
    else:
        out = _run("awk '/MemTotal/ {print $2*1024}' /proc/meminfo")
    return round(int(float(out)) / 1024 ** 3, 1) if out else None


def max_clock_ghz():
    try:
        import psutil
        f = psutil.cpu_freq()
        return round(f.max / 1000, 2) if f and f.max else None
    except ImportError:
        return None


def os_name():
    system = platform.system()
    if system == 'Windows':
        return f'Windows {platform.release()} (build {platform.version()})'
    if system == 'Darwin':
        return f'macOS {platform.mac_ver()[0]}'
    pretty = _run("grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '\"'")
    return pretty or f'Linux {platform.release()}'


def pkg(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return 'not installed'


if __name__ == '__main__':
    info = {
        'cpu_model': cpu_model(),
        'physical_cores': physical_cores(),
        'logical_threads': os.cpu_count(),
        'max_clock_GHz': max_clock_ghz(),
        'ram_GB': ram_gb(),
        'os': os_name(),
        'machine': platform.machine(),
        'python': sys.version.split()[0],
        'numpy': pkg('numpy'),
        'cvxpy': pkg('cvxpy'),
        'clarabel': pkg('clarabel'),
        'scs': pkg('scs'),
        'scipy': pkg('scipy'),
        'matplotlib': pkg('matplotlib'),
    }
    for k, v in info.items():
        print(f'{k:16s}: {v}')
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'environment.json')
    with open(out, 'w') as f:
        json.dump(info, f, indent=2)
    print(f'\nSaved to {os.path.normpath(out)}')
