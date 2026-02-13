"""
FITS Column Diagnostic Tool
----------------------------
Shows EXACTLY what columns exist in your FITS files.
"""

import sys
import os
import time
import traceback
import platform
import struct
import glob
from pathlib import Path

try:
    import numpy as np
    from astropy.io import fits
except ImportError as e:
    print("\nCRITICAL ERROR: Missing Required Libraries.")
    print(f"Details: {e}")
    print("Please run: pip install numpy astropy")
    input("Press Enter to exit...")
    sys.exit(1)

# ==============================================================================
# UTILITY FUNCTIONS - COPIED EXACTLY FROM convert_to_raw.py
# ==============================================================================

def force_pause():
    """
    Prevents the console window from closing immediately.
    Waits for user input before exiting.
    """
    print("\n" + "="*80)
    print("EXECUTION FINISHED.")
    print("="*80)
    try:
        input("Press ENTER to close this window...")
    except KeyboardInterrupt:
        pass

def log(message, level="INFO"):
    """
    Prints a formatted log message with timestamp.
    """
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{timestamp}] [{level:<5}] {message}")

def set_working_directory():
    """
    Forces the script to run in its own directory.
    Fixes the 'System32' double-click issue.
    """
    try:
        if getattr(sys, 'frozen', False):
            # If compiled as an exe
            script_path = os.path.dirname(sys.executable)
        else:
            # Standard python script
            script_path = os.path.dirname(os.path.abspath(__file__))
        
        os.chdir(script_path)
        log(f"Working directory set to: {script_path}", "SETUP")
        return script_path
    except Exception as e:
        log(f"Failed to set working directory: {e}", "ERROR")
        return os.getcwd()

def find_all_fits_files(directory="."):
    """
    Scans the directory for all .fits and .fit files.
    Returns sorted list of filenames (deduplicated).
    """
    fits_extensions = ['*.fits', '*.FITS', '*.fit', '*.FIT', '*.fits.gz', '*.FITS.gz']
    all_fits = []
    
    for ext in fits_extensions:
        pattern = os.path.join(directory, ext)
        found = glob.glob(pattern)
        all_fits.extend(found)
    
    # Get just the filenames and remove duplicates (case-insensitive filesystems)
    fits_files = sorted(list(set([os.path.basename(f) for f in all_fits])))
    
    return fits_files

# ==============================================================================
# MAIN DIAGNOSTIC
# ==============================================================================

def main():
    try:
        print("\n" + "#" * 80)
        print("   FITS DIAGNOSTIC TOOL")
        print("#" * 80 + "\n")

        # 1. Setup Environment - SAME AS convert_to_raw.py
        current_dir = set_working_directory()
        
        # 2. Find all FITS files - SAME AS convert_to_raw.py
        fits_files = find_all_fits_files(current_dir)
        
        if not fits_files:
            log("No FITS files found in current directory.", "ERROR")
            print("\nSearched for: *.fits, *.FITS, *.fit, *.FIT, *.fits.gz, *.FITS.gz")
            print("Please place .fits files in the same folder as this script.")
            return
        
        log(f"Found {len(fits_files)} FITS file(s)", "INFO")
        
        # 3. Select file
        if len(sys.argv) >= 2:
            fits_file = sys.argv[1]
            log(f"Using command line argument: {fits_file}", "INFO")
        else:
            print("\nAvailable FITS files:")
            for i, f in enumerate(fits_files, 1):
                size_mb = os.path.getsize(f) / (1024*1024)
                print(f"  [{i}] {f} ({size_mb:.2f} MB)")
            
            if len(fits_files) == 1:
                fits_file = fits_files[0]
                log(f"Auto-selecting only file: {fits_file}", "INFO")
            else:
                choice = input(f"\nSelect file number (1-{len(fits_files)}): ").strip()
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(fits_files):
                        fits_file = fits_files[idx]
                    else:
                        print("Invalid selection.")
                        return
                except ValueError:
                    print("Invalid input.")
                    return
        
        # 4. Run diagnostic
        print("\n" + "="*80)
        print(f"FITS FILE DIAGNOSTIC: {fits_file}")
        print("="*80)

        with fits.open(fits_file, memmap=True) as hdul:
            print(f"\nTotal HDUs (Header Data Units): {len(hdul)}")
            print("\n" + "="*80)
            
            all_columns_found = []
            
            for i, hdu in enumerate(hdul):
                print(f"\n--- HDU {i} ---")
                print(f"Type: {type(hdu).__name__}")
                
                # Show header info
                if hasattr(hdu, 'header'):
                    if 'EXTNAME' in hdu.header:
                        print(f"Extension Name: {hdu.header['EXTNAME']}")
                    if 'NAXIS' in hdu.header:
                        print(f"NAXIS: {hdu.header['NAXIS']}")
                
                # Binary Table
                if hasattr(hdu, 'columns') and hdu.columns:
                    print(f"\n  Binary Table with {len(hdu.columns.names)} columns:")
                    print(f"  {'Column Name':<25} {'Format':<10} {'Shape':<20} {'Bytes':<15}")
                    print(f"  {'-'*25} {'-'*10} {'-'*20} {'-'*15}")
                    
                    for col_name in hdu.columns.names:
                        all_columns_found.append(col_name)
                        col = hdu.columns[col_name]
                        
                        # Get actual data info
                        try:
                            data = hdu.data[col_name]
                            shape = data.shape
                            dtype = data.dtype
                            size_bytes = data.size * data.dtype.itemsize
                            size_mb = size_bytes / (1024*1024)
                            
                            print(f"  {col_name:<25} {col.format:<10} {str(shape):<20} {size_mb:>10.2f} MB")
                        except Exception as e:
                            print(f"  {col_name:<25} {col.format:<10} {'ERROR':<20} {str(e)}")
                
                # Image data
                elif hdu.data is not None:
                    if i == 0 and hdu.data.size < 2:
                        print("  Empty Primary HDU (no data)")
                    else:
                        print(f"\n  Image Data:")
                        print(f"    Shape: {hdu.data.shape}")
                        print(f"    dtype: {hdu.data.dtype}")
                        size_mb = (hdu.data.size * hdu.data.dtype.itemsize) / (1024*1024)
                        print(f"    Size: {size_mb:.2f} MB")
                else:
                    print("  No data (header only)")
            
            # Summary
            print("\n" + "="*80)
            print("COLUMN SUMMARY")
            print("="*80)
            print(f"\nTotal columns found: {len(all_columns_found)}")
            
            if all_columns_found:
                print("\nAll column names (sorted):")
                for name in sorted(all_columns_found):
                    print(f"  - {name}")
                
                # Check against expected Planck columns
                expected_planck = {
                    'Stokes': ['I_STOKES', 'Q_STOKES', 'U_STOKES'],
                    'Coverage': ['HITS'],
                    'Covariances': ['II_COV', 'QQ_COV', 'UU_COV', 'IQ_COV', 'IU_COV', 'QU_COV'],
                    'Variances': ['I_VAR', 'Q_VAR', 'U_VAR'],
                }
                
                print("\n" + "="*80)
                print("EXPECTED PLANCK COLUMNS CHECK")
                print("="*80)
                
                for category, expected in expected_planck.items():
                    found = [col for col in expected if col in all_columns_found]
                    missing = [col for col in expected if col not in all_columns_found]
                    
                    print(f"\n{category}:")
                    if found:
                        for col in found:
                            print(f"  ✓ {col} - FOUND")
                    if missing:
                        for col in missing:
                            print(f"  ✗ {col} - NOT IN THIS FILE")
                
                # Additional columns not in expected list
                expected_all = []
                for cols in expected_planck.values():
                    expected_all.extend(cols)
                
                additional = [col for col in all_columns_found if col not in expected_all]
                if additional:
                    print(f"\nAdditional columns (not in standard list):")
                    for col in additional:
                        print(f"  + {col}")
            
            print("\n" + "="*80)
            print("\nDIAGNOSTIC COMPLETE")
            print("="*80)
            
            if len(all_columns_found) == 0:
                print("\n⚠️  WARNING: NO COLUMNS FOUND!")
                print("This file may be corrupted or empty.")
            elif len(all_columns_found) < 3:
                print("\n⚠️  WARNING: Very few columns found!")
                print("Check if this is the correct file.")
            elif len(all_columns_found) < 7:
                print("\n⚠️  NOTE: Fewer columns than expected for full Planck data")
                print("Likely reasons:")
                print("  - Component-separated map (covariances removed) - THIS IS NORMAL")
                print("  - Lower frequency data (30-70 GHz lacks Q/U)")
                print("  - Simplified/processed map")
            else:
                print("\n✓ Column count looks good for full Planck frequency data")
            
            print(f"\n{len(all_columns_found)} columns will be extracted when you run convert_to_raw")

    except FileNotFoundError as e:
        log(f"File not found: {e}", "ERROR")
    except Exception as e:
        log("UNEXPECTED ERROR", "CRASH")
        traceback.print_exc()

if __name__ == "__main__":
    main()
    force_pause()