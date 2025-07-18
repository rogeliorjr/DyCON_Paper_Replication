import os
import h5py
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom
import glob


def normalize_image(image):
    """Normalize image to [0, 1] range using z-score + min-max normalization"""
    image = image.astype(np.float32)

    # Handle edge cases
    if np.all(image == 0):
        return image

    # Z-score normalization (per volume, only non-zero voxels)
    nonzero_mask = image > 0
    if np.any(nonzero_mask):
        mean = np.mean(image[nonzero_mask])
        std = np.std(image[nonzero_mask])
        if std > 0:
            image = np.where(nonzero_mask, (image - mean) / std, 0)

    # Min-max normalization to [0, 1]
    image_min = np.min(image)
    image_max = np.max(image)

    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)

    return image


def find_isles_bids_files(base_dir, case_name, modality='dwi'):
    """
    Find image and mask files for ISLES22 BIDS format

    Args:
        base_dir: Base directory containing ISLES22 data
        case_name: Case name (e.g., 'sub-strokecase0001')
        modality: Image modality to use ('dwi', 'adc', or 'flair')

    Returns:
        tuple: (image_path, mask_path)
    """

    # Find image file
    image_path = None

    # Look in dwi folder first
    if modality.lower() == 'dwi':
        dwi_pattern = os.path.join(base_dir, case_name, 'ses-0001', 'dwi', f'{case_name}_ses-0001_dwi.nii.gz')
        if os.path.exists(dwi_pattern):
            image_path = dwi_pattern

    elif modality.lower() == 'adc':
        adc_pattern = os.path.join(base_dir, case_name, 'ses-0001', 'dwi', f'{case_name}_ses-0001_adc.nii.gz')
        if os.path.exists(adc_pattern):
            image_path = adc_pattern

    elif modality.lower() == 'flair':
        flair_pattern = os.path.join(base_dir, case_name, 'ses-0001', 'anat', f'{case_name}_ses-0001_FLAIR.nii.gz')
        if os.path.exists(flair_pattern):
            image_path = flair_pattern

    # If primary modality not found, try others as fallback
    if image_path is None:
        # Try DWI first (most common for stroke detection)
        for fallback_modality in ['dwi', 'adc', 'flair']:
            if fallback_modality == modality.lower():
                continue  # Skip already tried modality

            if fallback_modality == 'dwi':
                fallback_path = os.path.join(base_dir, case_name, 'ses-0001', 'dwi', f'{case_name}_ses-0001_dwi.nii.gz')
            elif fallback_modality == 'adc':
                fallback_path = os.path.join(base_dir, case_name, 'ses-0001', 'dwi', f'{case_name}_ses-0001_adc.nii.gz')
            elif fallback_modality == 'flair':
                fallback_path = os.path.join(base_dir, case_name, 'ses-0001', 'anat',
                                             f'{case_name}_ses-0001_FLAIR.nii.gz')

            if os.path.exists(fallback_path):
                image_path = fallback_path
                print(f"  Using {fallback_modality.upper()} as fallback for {case_name}")
                break

    # Find mask file in derivatives
    mask_path = os.path.join(base_dir, 'derivatives', case_name, 'ses-0001', f'{case_name}_ses-0001_msk.nii.gz')

    if not os.path.exists(mask_path):
        mask_path = None

    return image_path, mask_path


def get_case_list(base_dir):
    """Get list of all available cases"""
    cases = []

    # Look for case directories
    for item in os.listdir(base_dir):
        if item.startswith('sub-strokecase') and os.path.isdir(os.path.join(base_dir, item)):
            cases.append(item)

    cases.sort()
    return cases


def process_isles_bids_case(base_dir, case_name, output_path, modality='dwi'):
    """Process a single ISLES22 BIDS case and save as h5 file"""

    try:
        # Find image and mask files
        image_path, mask_path = find_isles_bids_files(base_dir, case_name, modality)

        if image_path is None:
            print(f"❌ No suitable image modality found for {case_name}")
            return False

        if mask_path is None:
            print(f"❌ No mask file found for {case_name}")
            return False

        # Load image and mask
        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)

        # Get data
        image_data = image_nii.get_fdata()
        mask_data = mask_nii.get_fdata()

        # Check if data is valid
        if image_data.size == 0 or mask_data.size == 0:
            print(f"❌ Empty data for {case_name}")
            return False

        # Print original shape for debugging
        print(f"  Original shape: {image_data.shape}")

        # Normalize image
        image_data = normalize_image(image_data)

        # Convert mask to binary (lesion vs background)
        mask_data = (mask_data > 0.5).astype(np.float64)

        # Target shape for ISLES22: (112, 112, 64)
        target_shape = (112, 112, 64)

        # Calculate zoom factors
        zoom_factors = [t / s for t, s in zip(target_shape, image_data.shape)]

        # Resize image and mask
        image_resized = zoom(image_data, zoom_factors, order=1)  # Linear interpolation for image
        mask_resized = zoom(mask_data, zoom_factors, order=0)  # Nearest neighbor for mask

        # Ensure correct types and shapes
        image_resized = image_resized.astype(np.float64)
        mask_resized = mask_resized.astype(np.float64)

        # Ensure exact target shape (due to floating point precision)
        if image_resized.shape != target_shape:
            # Simple center crop/pad to exact target shape
            def resize_to_exact_shape(data, target_shape):
                current_shape = data.shape
                result = np.zeros(target_shape, dtype=data.dtype)

                # Calculate center position for each dimension
                for i in range(3):
                    current_size = current_shape[i]
                    target_size = target_shape[i]

                    if current_size >= target_size:
                        # Crop from center
                        start = (current_size - target_size) // 2
                        if i == 0:
                            result[:, :, :] = data[start:start + target_size, :, :]
                        elif i == 1:
                            result[:, :, :] = data[:, start:start + target_size, :]
                        else:
                            result[:, :, :] = data[:, :, start:start + target_size]
                    else:
                        # Pad to center
                        start = (target_size - current_size) // 2
                        if i == 0:
                            result[start:start + current_size, :, :] = data
                        elif i == 1:
                            result[:, start:start + current_size, :] = data
                        else:
                            result[:, :, start:start + current_size] = data

                return result

            # Apply exact resizing if needed
            if image_resized.shape != target_shape:
                image_resized = resize_to_exact_shape(image_resized, target_shape)
                mask_resized = resize_to_exact_shape(mask_resized, target_shape)

        # Extract case number for filename
        case_number = case_name.replace('sub-strokecase', '').zfill(3)
        h5_filename = f"case_{case_number}.h5"
        h5_path = os.path.join(output_path, h5_filename)

        # Get modality used
        modality_used = os.path.basename(image_path).split('_')[-1].replace('.nii.gz', '')

        # Save as h5 file
        with h5py.File(h5_path, 'w') as h5f:
            h5f.create_dataset('image', data=image_resized, compression='gzip')
            h5f.create_dataset('mask', data=mask_resized, compression='gzip')
            h5f.attrs['original_shape'] = image_data.shape
            h5f.attrs['case_name'] = case_name
            h5f.attrs['case_number'] = case_number
            h5f.attrs['modality'] = modality_used
            h5f.attrs['image_file'] = os.path.basename(image_path)
            h5f.attrs['mask_file'] = os.path.basename(mask_path)

        print(f"✅ {case_name}: {image_data.shape} -> {target_shape} ({modality_used})")
        return True

    except Exception as e:
        print(f"❌ Error processing {case_name}: {str(e)}")
        return False


def create_split_files(cases, output_dir, train_ratio=0.8):
    """Create train/val split files for ISLES22"""

    print(f"Found {len(cases)} cases")

    # Split into train/val
    np.random.seed(42)  # For reproducible splits
    indices = np.random.permutation(len(cases))

    train_size = int(len(cases) * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_cases = [cases[i] for i in train_indices]
    val_cases = [cases[i] for i in val_indices]

    # Convert to case numbers for the list files
    train_case_numbers = [case.replace('sub-strokecase', '').zfill(3) for case in train_cases]
    val_case_numbers = [case.replace('sub-strokecase', '').zfill(3) for case in val_cases]

    # Write split files
    train_file = os.path.join(output_dir, 'train.list')
    val_file = os.path.join(output_dir, 'val.list')

    with open(train_file, 'w') as f:
        for case_num in train_case_numbers:
            f.write(f"case_{case_num}\n")

    with open(val_file, 'w') as f:
        for case_num in val_case_numbers:
            f.write(f"case_{case_num}\n")

    print(f"Created split files:")
    print(f"  Train: {len(train_cases)} cases -> {train_file}")
    print(f"  Val: {len(val_cases)} cases -> {val_file}")

    return train_cases, val_cases


def preprocess_isles22_bids(input_dir, output_dir, modality='dwi', process_cases=None):
    """
    Main preprocessing function for ISLES22 BIDS format

    Args:
        input_dir: Path to ISLES22 BIDS directory
        output_dir: Output directory for H5 files
        modality: Image modality to use ('dwi', 'adc', 'flair')
        process_cases: List of specific cases to process (None = all cases)
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("ISLES22 BIDS DATA PREPROCESSING")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Primary modality: {modality.upper()}")
    print("=" * 80)

    # Get all available cases
    all_cases = get_case_list(input_dir)

    if not all_cases:
        print("❌ No cases found! Check your input directory path.")
        return

    # Determine which cases to process
    if process_cases is None:
        cases_to_process = all_cases
        # Create train/val splits
        train_cases, val_cases = create_split_files(all_cases, output_dir)
    else:
        cases_to_process = [case for case in all_cases if case in process_cases]
        print(f"Processing specific cases: {len(cases_to_process)} out of {len(all_cases)}")

    print(f"Processing {len(cases_to_process)} cases...")
    print("=" * 80)

    successful = 0
    failed = 0

    # Process cases
    for case_name in tqdm(cases_to_process, desc="Processing ISLES22 cases"):
        if process_isles_bids_case(input_dir, case_name, output_dir, modality):
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 80)
    print("PREPROCESSING SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 H5 files saved to: {output_dir}")
    print(f"🔧 Primary modality used: {modality.upper()}")

    # Show some examples of created files
    h5_files = [f for f in os.listdir(output_dir) if f.endswith('.h5')]
    if h5_files:
        print(f"\n📋 Example H5 files created:")
        for h5_file in sorted(h5_files)[:5]:
            print(f"  📄 {h5_file}")
        if len(h5_files) > 5:
            print(f"  ... and {len(h5_files) - 5} more files")


if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS
    RAW_DATA_DIR = "/home/dojo/PycharmProjects/DyCON_Paper_Replication/code/ISLES-2022"  # Your ISLES22 BIDS directory
    OUTPUT_DIR = "../data/ISLES22"  # Output directory for H5 files
    MODALITY = "dwi"  # Primary modality: 'dwi', 'adc', or 'flair'

    print("🔄 Processing ISLES22 BIDS dataset...")
    preprocess_isles22_bids(RAW_DATA_DIR, OUTPUT_DIR, MODALITY)

    print("\n🏁 ISLES22 preprocessing completed!")
    print("\nNext steps:")
    print("1. Verify the H5 files in the output directory")
    print("2. Check train.list and val.list files")
    print("3. Update your training script to use the new data path")