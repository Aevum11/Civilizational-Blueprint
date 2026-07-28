"""
CMB FITS to RAW Converter - Ultimate Edition
--------------------------------------------
Features:
1. Auto-detects execution environment (System32 vs Local).
2. scans FITS headers for I_STOKES, TEMPERATURE, I, or SIGNAL.
3. Enforces float32 (standard for raw binary processing).
4. Prevents window closure on both success and failure.
5. Verbose logging for debugging.
6. Memory mapping for large files.
"""

import sys
import os
import time
import traceback
import platform
import struct

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
# CONFIGURATION & CONSTANTS
# ==============================================================================

DEFAULT_INPUT_FILE = "COM_CMB_IQU-smica_2048_R3.00_full.fits"
DEFAULT_OUTPUT_FILE = "pure_cmb_intensity.bin"

# Column names to search for, in order of priority
TARGET_COLUMNS = [
    'I_STOKES',    # Standard Planck
    'TEMPERATURE', # Common alternative
    'I',           # Generic Intensity
    'TEMP',        # Short form
    'SIGNAL',      # Processed maps
    'INTENSITY'    # Generic
]

# ==============================================================================
# UTILITY FUNCTIONS
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

# ==============================================================================
# CORE CONVERTER CLASS
# ==============================================================================

class CMBRawExtractor:
    def __init__(self, input_filename, output_filename):
        self.input_file = os.path.abspath(input_filename)
        self.output_file = os.path.abspath(output_filename)
        self.data_array = None
        self.header_info = {}

    def validate_environment(self):
        """Checks if input file exists."""
        if not os.path.exists(self.input_file):
            log(f"Input file not found: {self.input_file}", "FATAL")
            log("Please ensure the .fits file is in the same folder.", "HINT")
            raise FileNotFoundError("Input file missing.")
        
        log(f"Input File:  {self.input_file}", "CONFIG")
        log(f"Output File: {self.output_file}", "CONFIG")

    def read_fits(self):
        """Opens the FITS file and locates the intensity data."""
        log("Opening FITS file (memmap=True)...", "ACTION")
        
        with fits.open(self.input_file, memmap=True) as hdul:
            log(f"FITS file opened. Found {len(hdul)} extensions.", "INFO")
            
            # Print structure for debugging
            # Capture info string for logging
            # hdul.info() prints to stdout, we let it pass through
            print("-" * 50)
            hdul.info()
            print("-" * 50)

            found_data = None
            found_name = None

            # Iterate through all HDUs (Header Data Units)
            for i, hdu in enumerate(hdul):
                
                # Check 1: Binary Table (Standard for HEALPix/Planck)
                if hasattr(hdu, 'columns'):
                    col_names = hdu.columns.names
                    log(f"Extension {i} is a Table. Columns: {col_names}", "DEBUG")
                    
                    # Search for priority columns
                    for target in TARGET_COLUMNS:
                        if target in col_names:
                            log(f"--> MATCH FOUND: Column '{target}' in Ext {i}", "SUCCESS")
                            found_data = hdu.data[target]
                            found_name = target
                            break
                    
                    if found_data is not None:
                        break

                # Check 2: Image Extension (Direct Array)
                elif hdu.data is not None:
                    # Skip PrimaryHDU if it is empty/1D 
                    if i == 0 and hdu.data.size < 2:
                        continue
                        
                    log(f"Extension {i} is an Image Array (Shape: {hdu.data.shape})", "DEBUG")
                    # If we haven't found a table column yet, this might be it
                    # But we usually prefer Tables for Planck data.
                    # We will treat this as a fallback.
                    if found_data is None:
                        log(f"--> Using Extension {i} Image Data as fallback.", "WARN")
                        found_data = hdu.data
                        found_name = "IMAGE_HDU"
                        break

            if found_data is None:
                log("Could not find any recognized Intensity data.", "FATAL")
                log(f"Searched for: {TARGET_COLUMNS}", "INFO")
                raise ValueError("No valid data column found.")

            # Store the data
            self.data_array = found_data
            log(f"Data extracted successfully from source '{found_name}'.", "INFO")

    def process_data(self):
        """Flattens array and enforces float32."""
        if self.data_array is None:
            raise ValueError("No data to process.")

        log("Processing raw array...", "ACTION")
        
        # 1. Flatten multidimensional arrays (HEALPix is 1D, but logic safety)
        if self.data_array.ndim > 1:
            log(f"Flattening array from shape {self.data_array.shape}...", "INFO")
            self.data_array = self.data_array.flatten()
        
        # 2. Check Data Type
        original_dtype = self.data_array.dtype
        log(f"Original Data Type: {original_dtype}", "DEBUG")
        
        # 3. Convert to float32 (Standard for raw graphics/binary)
        if self.data_array.dtype != np.float32:
            log("Converting data to float32...", "INFO")
            self.data_array = self.data_array.astype(np.float32)
        
        # 4. Statistics
        d_min = np.nanmin(self.data_array)
        d_max = np.nanmax(self.data_array)
        d_mean = np.nanmean(self.data_array)
        
        log(f"Stats: Min={d_min:.5f}, Max={d_max:.5f}, Mean={d_mean:.5f}", "STATS")
        log(f"Total Pixels: {self.data_array.size}", "STATS")

    def save_raw(self):
        """Writes the raw bytes to disk."""
        log(f"Writing raw bytes to: {self.output_file}", "ACTION")
        
        try:
            self.data_array.tofile(self.output_file)
        except IOError as e:
            log(f"Disk Write Error: {e}", "FATAL")
            raise

    def verify(self):
        """Verifies the output file exists and has size."""
        if os.path.exists(self.output_file):
            size_bytes = os.path.getsize(self.output_file)
            size_mb = size_bytes / (1024 * 1024)
            log(f"Verification Successful.", "SUCCESS")
            log(f"Final File Size: {size_mb:.2f} MB", "INFO")
            log(f"Expected Size: {(self.data_array.size * 4) / (1024*1024):.2f} MB", "DEBUG")
        else:
            log("File verification failed. File not found on disk.", "FATAL")

# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================

def main():
    try:
        print("\n" + "#" * 60)
        print("   ASTROPY FITS TO RAW CONVERTER - ROBUST EDITION")
        print("#" * 60 + "\n")

        # 1. Setup Environment
        current_dir = set_working_directory()

        # 2. Parse Arguments (or use defaults)
        if len(sys.argv) >= 3:
            in_file = sys.argv[1]
            out_file = sys.argv[2]
            log("Using provided command line arguments.", "INIT")
        else:
            in_file = DEFAULT_INPUT_FILE
            out_file = DEFAULT_OUTPUT_FILE
            log("No arguments provided. Using internal defaults.", "INIT")
            log(f"Default Input: {in_file}", "INIT")

        # 3. Initialize Extractor
        extractor = CMBRawExtractor(in_file, out_file)

        # 4. Run Pipeline
        extractor.validate_environment()
        extractor.read_fits()
        extractor.process_data()
        extractor.save_raw()
        extractor.verify()
        
        print("\n" + "*" * 60)
        print("CONVERSION COMPLETE")
        print("You can now import this .bin file as raw float32.")
        print("*" * 60)

    except Exception as e:
        print("\n" + "!" * 60)
        log("UNEXPECTED ERROR OCCURRED", "CRASH")
        print("!" * 60)
        traceback.print_exc()
        print("!" * 60)

if __name__ == "__main__":
    main()
    force_pause()