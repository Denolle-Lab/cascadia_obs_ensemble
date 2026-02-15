#!/usr/bin/env python3
"""
Environment Verification Script for Cascadia OBS Ensemble Project

This script verifies that all required packages are installed and functional.
Run this after setting up your environment to ensure everything is working.

Usage:
    python verify_environment.py
"""

import sys
from importlib.metadata import version, PackageNotFoundError

# ANSI color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{text:^70}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

def check_import(module_name, package_name=None, from_module=None):
    """
    Attempt to import a module and report status.
    
    Args:
        module_name: Name to import
        package_name: Package name for version checking (if different from module)
        from_module: If provided, do "from from_module import module_name"
    """
    if package_name is None:
        package_name = module_name.split('.')[0]
    
    try:
        if from_module:
            exec(f"from {from_module} import {module_name}")
        else:
            __import__(module_name)
        
        try:
            ver = version(package_name)
            print(f"  {GREEN}✓{RESET} {module_name:30s} (v{ver})")
            return True, ver
        except PackageNotFoundError:
            print(f"  {GREEN}✓{RESET} {module_name:30s} (version unknown)")
            return True, None
    except ImportError as e:
        print(f"  {RED}✗{RESET} {module_name:30s} - MISSING")
        print(f"      Error: {e}")
        return False, None
    except Exception as e:
        print(f"  {YELLOW}⚠{RESET} {module_name:30s} - ERROR")
        print(f"      Error: {e}")
        return False, None

def main():
    print_header("Cascadia OBS Ensemble - Environment Verification")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}\n")
    
    results = {}
    
    # Core Scientific Computing
    print_header("Core Scientific Computing")
    results['numpy'] = check_import('numpy')
    results['pandas'] = check_import('pandas')
    results['scipy'] = check_import('scipy')
    results['matplotlib'] = check_import('matplotlib')
    
    # Seismology Packages
    print_header("Seismology Packages")
    results['obspy'] = check_import('obspy')
    results['seisbench'] = check_import('seisbench')
    
    # Check specific obspy submodules
    print("\n  Checking ObsPy submodules:")
    check_import('obspy.clients.fdsn', 'obspy')
    check_import('obspy.core.utcdatetime', 'obspy')
    check_import('obspy.signal.trigger', 'obspy')
    check_import('obspy.geodetics', 'obspy')
    
    # Machine Learning
    print_header("Machine Learning")
    results['torch'] = check_import('torch')
    
    # Test PyTorch device availability
    if results['torch'][0]:
        try:
            import torch
            print(f"\n  PyTorch device info:")
            print(f"    CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"    CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"    Default device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        except Exception as e:
            print(f"  {YELLOW}⚠{RESET} Could not check PyTorch device: {e}")
    
    # Parallel Computing
    print_header("Parallel Computing")
    results['dask'] = check_import('dask')
    
    # Check dask submodules
    if results['dask'][0]:
        print("\n  Checking Dask submodules:")
        check_import('dask.delayed', 'dask')
        check_import('dask.diagnostics', 'dask')
    
    # Data I/O
    print_header("Data I/O and Utilities")
    results['h5py'] = check_import('h5py')
    results['tqdm'] = check_import('tqdm')
    results['openpyxl'] = check_import('openpyxl')
    results['adjustText'] = check_import('adjustText')
    
    # Visualization
    print_header("Visualization")
    results['basemap'] = check_import('mpl_toolkits.basemap', 'basemap', 'mpl_toolkits')
    
    # Special/GitHub Packages
    print_header("Special Packages (GitHub installs)")
    results['ELEP'] = check_import('ELEP')
    
    # Check ELEP submodules
    if results['ELEP'][0]:
        print("\n  Checking ELEP submodules:")
        check_import('ensemble_statistics', 'ELEP', 'ELEP.elep')
        check_import('ensemble_coherence', 'ELEP', 'ELEP.elep')
        check_import('trigger_func', 'ELEP', 'ELEP.elep')
    
    results['pnwstore'] = check_import('pnwstore')
    
    # Check pnwstore submodule
    if results['pnwstore'][0]:
        print("\n  Checking pnwstore submodule:")
        wfc_status = check_import('WaveformClient', 'pnwstore', 'pnwstore.mseed')
        if wfc_status[0]:
            try:
                from pnwstore.mseed import WaveformClient
                client = WaveformClient()
                print(f"      {GREEN}✓{RESET} WaveformClient initialized successfully")
            except Exception as e:
                print(f"      {YELLOW}⚠{RESET} WaveformClient initialization failed: {e}")
    
    # Optional packages
    print_header("Optional Packages")
    check_import('pyocto')
    
    # Summary
    print_header("Summary")
    
    total = len(results)
    passed = sum(1 for status, _ in results.values() if status)
    failed = total - passed
    
    print(f"Total packages checked: {total}")
    print(f"{GREEN}Passed: {passed}{RESET}")
    if failed > 0:
        print(f"{RED}Failed: {failed}{RESET}")
    
    # Critical packages check
    critical_packages = ['numpy', 'pandas', 'obspy', 'seisbench', 'torch', 
                        'dask', 'ELEP', 'pnwstore']
    
    missing_critical = [pkg for pkg in critical_packages 
                       if pkg in results and not results[pkg][0]]
    
    print("\n" + "="*70)
    
    if missing_critical:
        print(f"\n{RED}✗ CRITICAL PACKAGES MISSING:{RESET}")
        for pkg in missing_critical:
            print(f"  - {pkg}")
        print(f"\n{YELLOW}Installation instructions:{RESET}")
        if 'ELEP' in missing_critical:
            print(f"  ELEP: pip install git+https://github.com/congcy/ELEP.git")
        if 'pnwstore' in missing_critical:
            print(f"  pnwstore: pip install git+https://github.com/niyiyu/pnwstore.git")
        print(f"\nSee INSTALL.md for complete installation instructions.")
        return 1
    else:
        print(f"\n{GREEN}✓ All critical packages are installed!{RESET}")
        print(f"\nYour environment is ready to run:")
        print(f"  - Parallel picking scripts (1_picking/)")
        print(f"  - Event waveform processing (4_relocation/)")
        print(f"\nTo test the environment with real data, try:")
        print(f"  jupyter notebook 4_relocation/test_event_waveform_processing.ipynb")
        return 0

if __name__ == "__main__":
    sys.exit(main())
