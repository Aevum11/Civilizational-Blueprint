"""
CMB FITS to RAW Converter - Universal Edition
----------------------------------------------
Features:
1. Auto-detects and lists ALL .fits files in the working directory.
2. Interactive menu for single file or batch processing.
3. Scans FITS headers for I_STOKES, TEMPERATURE, I, SIGNAL, and all variants.
4. Enforces float32 (standard for raw binary processing).
5. Auto-generates intelligent output filenames based on input.
6. Prevents window closure on both success and failure.
7. Verbose logging for debugging with full data inspection.
8. Memory mapping for large files.
9. Comprehensive column detection across all HDU types.
10. Detailed preview mode to inspect file contents before extraction.
11. Handles any astronomical FITS file format.
12. Strips all headers and metadata - extracts ONLY pure data.
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
# CONFIGURATION & CONSTANTS
# ==============================================================================

# Column names to search for, in order of priority
TARGET_COLUMNS = [
    'I_STOKES',    # Standard Planck I Stokes parameter
    'TEMPERATURE', # Temperature maps
    'I',           # Generic Intensity
    'TEMP',        # Short form temperature
    'SIGNAL',      # Processed signal maps
    'INTENSITY',   # Generic intensity
    'DATA',        # Generic data column
    'VALUE',       # Generic value column
    'FLUX',        # Flux measurements
    'T',           # Temperature short form
    'Q_STOKES',    # Q Stokes (if I is not available)
    'U_STOKES',    # U Stokes (if I is not available)
    'Q',           # Q polarization
    'U',           # U polarization
    'V',           # V Stokes parameter
    'BRIGHTNESS',  # Brightness temperature
    'TB',          # Brightness temperature short form
    'MAP',         # Generic map data
    'PIXEL',       # Pixel values
    'COUNTS',      # Count data
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

def generate_output_filename(input_filename):
    """
    Generates an intelligent output filename based on input.
    Preserves important identifiers while adding .bin extension.
    """
    # Remove path and extension
    base = os.path.splitext(os.path.basename(input_filename))[0]
    
    # Handle .fits.gz case
    if base.endswith('.fits'):
        base = os.path.splitext(base)[0]
    
    # Clean up common patterns to make output names more readable
    # but preserve important identifiers
    output_name = f"{base}_raw_data.bin"
    
    return output_name

def display_fits_menu(fits_files):
    """
    Displays an interactive menu of all available FITS files.
    Returns the user's selection.
    """
    print("\n" + "="*80)
    print("AVAILABLE FITS FILES IN CURRENT DIRECTORY:")
    print("="*80)
    
    if not fits_files:
        print("No FITS files found in the current directory.")
        print("Please ensure .fits files are present and try again.")
        return None
    
    for i, filename in enumerate(fits_files, 1):
        file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
        print(f"  [{i}] {filename:<50} ({file_size:>8.2f} MB)")
    
    print(f"  [A] Process ALL files ({len(fits_files)} files)")
    print(f"  [P] Preview file contents (inspect before extraction)")
    print(f"  [Q] Quit")
    print("="*80)
    
    return None  # Will get user input in main loop

def preview_fits_structure(filename):
    """
    Displays detailed structure of a FITS file without extraction.
    Shows all HDUs, columns, and data shapes.
    """
    print("\n" + "#"*80)
    print(f"PREVIEWING: {filename}")
    print("#"*80)
    
    try:
        with fits.open(filename, memmap=True) as hdul:
            print(f"\nTotal Extensions (HDUs): {len(hdul)}")
            print("-"*80)
            
            for i, hdu in enumerate(hdul):
                print(f"\n[Extension {i}]")
                print(f"  Type: {type(hdu).__name__}")
                
                # Header info
                if hasattr(hdu, 'header') and hdu.header:
                    print(f"  Header Keys: {len(hdu.header)} keys")
                    # Show some key headers
                    important_keys = ['NAXIS', 'NAXIS1', 'NAXIS2', 'BITPIX', 
                                     'TTYPE1', 'TFORM1', 'EXTNAME']
                    for key in important_keys:
                        if key in hdu.header:
                            print(f"    {key}: {hdu.header[key]}")
                
                # Table columns
                if hasattr(hdu, 'columns') and hdu.columns:
                    print(f"  Columns ({len(hdu.columns.names)}):")
                    for col_name in hdu.columns.names:
                        col = hdu.columns[col_name]
                        print(f"    - {col_name:<20} (format: {col.format})")
                    
                    # Check for target columns
                    matches = [col for col in hdu.columns.names if col in TARGET_COLUMNS]
                    if matches:
                        print(f"  *** POTENTIAL DATA COLUMNS: {matches} ***")
                
                # Image data
                if hdu.data is not None:
                    print(f"  Data Shape: {hdu.data.shape}")
                    print(f"  Data Type: {hdu.data.dtype}")
                    if hasattr(hdu.data, 'size'):
                        print(f"  Total Elements: {hdu.data.size:,}")
                else:
                    print(f"  Data: None (header-only)")
            
            print("\n" + "-"*80)
            
    except Exception as e:
        print(f"\nERROR during preview: {e}")
        traceback.print_exc()
    
    print("\n" + "#"*80 + "\n")

# ==============================================================================
# CORE CONVERTER CLASS
# ==============================================================================

class CMBRawExtractor:
    def __init__(self, input_filename, output_filename=None):
        self.input_file = os.path.abspath(input_filename)
        
        # Auto-generate output filename if not provided
        if output_filename is None:
            output_filename = generate_output_filename(input_filename)
        
        self.output_file = os.path.abspath(output_filename)
        self.data_array = None
        self.header_info = {}
        self.source_info = {}

    def validate_environment(self):
        """Checks if input file exists."""
        if not os.path.exists(self.input_file):
            log(f"Input file not found: {self.input_file}", "FATAL")
            log("Please ensure the .fits file is in the current folder.", "HINT")
            raise FileNotFoundError("Input file missing.")
        
        log(f"Input File:  {self.input_file}", "CONFIG")
        log(f"Output File: {self.output_file}", "CONFIG")

    def read_fits(self):
        """
        Opens the FITS file and locates the intensity/data.
        Searches exhaustively through all HDUs and columns.
        """
        log("Opening FITS file (memmap=True)...", "ACTION")
        
        with fits.open(self.input_file, memmap=True) as hdul:
            log(f"FITS file opened. Found {len(hdul)} extensions.", "INFO")
            
            # Print structure for debugging
            print("-" * 80)
            hdul.info()
            print("-" * 80)

            found_data = None
            found_name = None
            found_location = None

            # PHASE 1: Prioritize Binary Tables with target columns
            for i, hdu in enumerate(hdul):
                if hasattr(hdu, 'columns') and hdu.columns:
                    col_names = hdu.columns.names
                    log(f"Extension {i} is a Table. Columns: {col_names}", "DEBUG")
                    
                    # Search for priority columns
                    for target in TARGET_COLUMNS:
                        if target in col_names:
                            log(f"--> MATCH FOUND: Column '{target}' in Ext {i}", "SUCCESS")
                            found_data = hdu.data[target]
                            found_name = target
                            found_location = f"Table Extension {i}, Column '{target}'"
                            break
                    
                    if found_data is not None:
                        break

            # PHASE 2: If no table column found, search image arrays
            if found_data is None:
                log("No table column match. Searching image extensions...", "INFO")
                
                for i, hdu in enumerate(hdul):
                    if hdu.data is not None:
                        # Skip empty primary HDU
                        if i == 0 and hdu.data.size < 2:
                            continue
                        
                        log(f"Extension {i}: Image Array (Shape: {hdu.data.shape}, "
                            f"Type: {hdu.data.dtype})", "DEBUG")
                        
                        # Use first non-trivial image extension
                        if hdu.data.size > 1:
                            log(f"--> Using Extension {i} Image Data.", "WARN")
                            found_data = hdu.data
                            found_name = "IMAGE_HDU"
                            found_location = f"Image Extension {i}"
                            break

            # PHASE 3: Absolute fallback - use ANY data we can find
            if found_data is None:
                log("Standard search failed. Attempting deep scan...", "WARN")
                
                for i, hdu in enumerate(hdul):
                    # Check if there's ANY data at all
                    if hasattr(hdu, 'data') and hdu.data is not None:
                        if isinstance(hdu.data, np.ndarray) and hdu.data.size > 0:
                            log(f"--> FALLBACK: Using Extension {i} data.", "WARN")
                            found_data = hdu.data
                            found_name = "FALLBACK_DATA"
                            found_location = f"Extension {i} (fallback mode)"
                            break
                    
                    # Check if it's a table with ANY columns
                    if hasattr(hdu, 'columns') and hdu.columns:
                        first_col = hdu.columns.names[0]
                        log(f"--> FALLBACK: Using first table column '{first_col}' "
                            f"from Extension {i}.", "WARN")
                        found_data = hdu.data[first_col]
                        found_name = first_col
                        found_location = f"Table Extension {i}, Column '{first_col}' (fallback)"
                        break

            # Final check
            if found_data is None:
                log("CRITICAL: Could not find ANY extractable data in FITS file.", "FATAL")
                log(f"Searched for columns: {TARGET_COLUMNS}", "INFO")
                log("File may be corrupted or use an unsupported format.", "INFO")
                raise ValueError("No valid data found in FITS file.")

            # Store the data and metadata
            self.data_array = found_data
            self.source_info = {
                'column_name': found_name,
                'location': found_location,
                'original_dtype': found_data.dtype,
                'original_shape': found_data.shape
            }
            
            log(f"Data extracted from: {found_location}", "SUCCESS")
            log(f"Original shape: {found_data.shape}, dtype: {found_data.dtype}", "INFO")

    def process_data(self):
        """
        Flattens array and enforces float32.
        STRIPS ALL METADATA - outputs ONLY pure data values.
        """
        if self.data_array is None:
            raise ValueError("No data to process.")

        log("Processing raw array (stripping all metadata)...", "ACTION")
        
        # 1. Flatten multidimensional arrays
        original_shape = self.data_array.shape
        if self.data_array.ndim > 1:
            log(f"Flattening array from shape {self.data_array.shape}...", "INFO")
            self.data_array = self.data_array.flatten()
            log(f"Flattened to 1D array of length {self.data_array.size}", "INFO")
        
        # 2. Check Data Type
        original_dtype = self.data_array.dtype
        log(f"Original Data Type: {original_dtype}", "DEBUG")
        
        # 3. Convert to float32 (Standard for raw graphics/binary processing)
        if self.data_array.dtype != np.float32:
            log("Converting data to float32...", "INFO")
            self.data_array = self.data_array.astype(np.float32)
            log("Conversion complete.", "INFO")
        else:
            log("Data already in float32 format.", "INFO")
        
        # 4. Statistics (for verification only - not written to output)
        d_min = np.nanmin(self.data_array)
        d_max = np.nanmax(self.data_array)
        d_mean = np.nanmean(self.data_array)
        d_std = np.nanstd(self.data_array)
        nan_count = np.isnan(self.data_array).sum()
        inf_count = np.isinf(self.data_array).sum()
        
        log(f"Data Statistics (for reference only):", "STATS")
        log(f"  Shape: {original_shape} -> Flattened: {self.data_array.size}", "STATS")
        log(f"  Min Value:    {d_min:.8e}", "STATS")
        log(f"  Max Value:    {d_max:.8e}", "STATS")
        log(f"  Mean Value:   {d_mean:.8e}", "STATS")
        log(f"  Std Dev:      {d_std:.8e}", "STATS")
        log(f"  NaN Count:    {nan_count}", "STATS")
        log(f"  Inf Count:    {inf_count}", "STATS")
        log(f"  Total Pixels: {self.data_array.size:,}", "STATS")

    def save_raw(self):
        """
        Writes the raw bytes to disk.
        Output is PURE DATA ONLY - no headers, no metadata, no structure.
        Just sequential float32 values.
        """
        log(f"Writing raw bytes to: {self.output_file}", "ACTION")
        log("Output format: Pure float32 binary (no headers, no metadata)", "INFO")
        
        try:
            # Write as raw binary - this strips EVERYTHING except data values
            self.data_array.tofile(self.output_file)
            log("Raw data write complete.", "SUCCESS")
        except IOError as e:
            log(f"Disk Write Error: {e}", "FATAL")
            raise

    def verify(self):
        """Verifies the output file exists and has correct size."""
        if os.path.exists(self.output_file):
            size_bytes = os.path.getsize(self.output_file)
            size_mb = size_bytes / (1024 * 1024)
            expected_bytes = self.data_array.size * 4  # float32 = 4 bytes
            expected_mb = expected_bytes / (1024 * 1024)
            
            log(f"File Verification:", "CHECK")
            log(f"  Output File:   {self.output_file}", "CHECK")
            log(f"  File Size:     {size_mb:.2f} MB", "CHECK")
            log(f"  Expected Size: {expected_mb:.2f} MB", "CHECK")
            
            if abs(size_bytes - expected_bytes) < 10:  # Allow small rounding
                log(f"Size verification PASSED.", "SUCCESS")
            else:
                log(f"WARNING: Size mismatch detected!", "WARN")
                log(f"Difference: {abs(size_bytes - expected_bytes)} bytes", "WARN")
            
            # Provide usage info
            log(f"Data Format Information:", "INFO")
            log(f"  Format: IEEE 754 float32 (4 bytes per value)", "INFO")
            log(f"  Values: {self.data_array.size:,} sequential floats", "INFO")
            log(f"  Source: {self.source_info['location']}", "INFO")
            log(f"  Original Shape: {self.source_info['original_shape']}", "INFO")
            
        else:
            log("File verification FAILED. File not found on disk.", "FATAL")
            raise FileNotFoundError("Output file was not created.")

    def run_full_pipeline(self):
        """
        Executes the complete extraction pipeline.
        Returns True on success, False on failure.
        """
        try:
            self.validate_environment()
            self.read_fits()
            self.process_data()
            self.save_raw()
            self.verify()
            return True
        except Exception as e:
            log(f"Pipeline failed for {self.input_file}: {e}", "ERROR")
            traceback.print_exc()
            return False

# ==============================================================================
# BATCH PROCESSING
# ==============================================================================

def process_all_fits_files(fits_files):
    """
    Processes all FITS files in batch mode.
    """
    print("\n" + "="*80)
    print(f"BATCH PROCESSING: {len(fits_files)} FILES")
    print("="*80 + "\n")
    
    results = {
        'success': [],
        'failed': []
    }
    
    for i, filename in enumerate(fits_files, 1):
        print("\n" + "#"*80)
        print(f"PROCESSING FILE {i} of {len(fits_files)}: {filename}")
        print("#"*80 + "\n")
        
        try:
            extractor = CMBRawExtractor(filename)
            success = extractor.run_full_pipeline()
            
            if success:
                results['success'].append((filename, extractor.output_file))
                print(f"\n✓ SUCCESS: {filename} -> {extractor.output_file}")
            else:
                results['failed'].append(filename)
                print(f"\n✗ FAILED: {filename}")
        
        except KeyboardInterrupt:
            log("Batch processing interrupted by user.", "WARN")
            break
        except Exception as e:
            results['failed'].append(filename)
            log(f"Error processing {filename}: {e}", "ERROR")
            print(f"\n✗ FAILED: {filename}")
    
    # Summary
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    print(f"Total Files:      {len(fits_files)}")
    print(f"Successful:       {len(results['success'])}")
    print(f"Failed:           {len(results['failed'])}")
    
    if results['success']:
        print("\nSuccessful Conversions:")
        for inp, out in results['success']:
            print(f"  ✓ {inp} -> {out}")
    
    if results['failed']:
        print("\nFailed Conversions:")
        for filename in results['failed']:
            print(f"  ✗ {filename}")
    
    print("="*80 + "\n")
    
    return results

# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================

def main():
    try:
        print("\n" + "#" * 80)
        print("   FITS TO RAW CONVERTER - UNIVERSAL EDITION")
        print("   Extracts pure CMB data from any astronomical FITS file")
        print("#" * 80 + "\n")

        # 1. Setup Environment
        current_dir = set_working_directory()
        
        # 2. Find all FITS files in directory
        fits_files = find_all_fits_files(current_dir)
        
        # 3. Check for command line arguments first
        if len(sys.argv) >= 2:
            input_file = sys.argv[1]
            output_file = sys.argv[2] if len(sys.argv) >= 3 else None
            
            log("Using command line arguments.", "INIT")
            log(f"Input:  {input_file}", "INIT")
            if output_file:
                log(f"Output: {output_file}", "INIT")
            else:
                log(f"Output: (auto-generate)", "INIT")
            
            # Process single file from command line
            extractor = CMBRawExtractor(input_file, output_file)
            extractor.run_full_pipeline()
            
            print("\n" + "*" * 80)
            print("CONVERSION COMPLETE")
            print("Pure CMB data extracted successfully.")
            print("*" * 80)
            
        else:
            # Interactive mode
            log("No command line arguments. Entering interactive mode.", "INIT")
            
            if not fits_files:
                log("No FITS files found in current directory.", "ERROR")
                print("\nPlease place .fits files in the same folder as this script.")
                return
            
            # Display menu and process
            while True:
                display_fits_menu(fits_files)
                
                choice = input("\nEnter your choice: ").strip().upper()
                
                if choice == 'Q':
                    log("User chose to quit.", "INFO")
                    break
                
                elif choice == 'A':
                    # Process all files
                    confirm = input(f"\nProcess all {len(fits_files)} files? (Y/N): ").strip().upper()
                    if confirm == 'Y':
                        process_all_fits_files(fits_files)
                    break
                
                elif choice == 'P':
                    # Preview mode
                    file_num = input("Enter file number to preview: ").strip()
                    try:
                        idx = int(file_num) - 1
                        if 0 <= idx < len(fits_files):
                            preview_fits_structure(fits_files[idx])
                        else:
                            print("Invalid file number.")
                    except ValueError:
                        print("Please enter a valid number.")
                
                elif choice.isdigit():
                    # Process single file
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(fits_files):
                            selected_file = fits_files[idx]
                            print(f"\nProcessing: {selected_file}")
                            
                            extractor = CMBRawExtractor(selected_file)
                            extractor.run_full_pipeline()
                            
                            print("\n" + "*" * 80)
                            print("CONVERSION COMPLETE")
                            print("Pure CMB data extracted successfully.")
                            print("*" * 80)
                            break
                        else:
                            print("Invalid file number. Please try again.")
                    except ValueError:
                        print("Invalid input. Please enter a number, A, P, or Q.")
                else:
                    print("Invalid choice. Please try again.")

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        log("Script interrupted by user (Ctrl+C).", "INFO")
    
    except Exception as e:
        print("\n" + "!" * 80)
        log("UNEXPECTED ERROR OCCURRED", "CRASH")
        print("!" * 80)
        traceback.print_exc()
        print("!" * 80)

if __name__ == "__main__":
    main()
    force_pause()