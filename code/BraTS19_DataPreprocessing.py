import os
import h5py
import numpy as np
import nibabel as nib
from tqdm import tqdm


def normalize_image(image):
    """Normalize image to [0, 1] range"""
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


def find_case_directory(base_dir, case_name):
    """Find case directory in HGG or LGG subdirectories"""

    # Check both HGG and LGG subdirectories
    for subdir in ['HGG', 'LGG']:
        case_path = os.path.join(base_dir, subdir, case_name)
        if os.path.exists(case_path):
            return case_path

    # Also check directly in base directory (fallback)
    case_path = os.path.join(base_dir, case_name)
    if os.path.exists(case_path):
        return case_path

    return None


def find_case_files(case_path, case_name):
    """Find the NIfTI files for a case"""

    if not os.path.exists(case_path):
        return None, None, None, None, None

    files = os.listdir(case_path)
    # Look for both .nii.gz and .nii files
    nii_files = [f for f in files if f.endswith('.nii.gz') or f.endswith('.nii')]

    if len(nii_files) == 0:
        print(f"No .nii or .nii.gz files found in {case_path}")
        return None, None, None, None, None

    # First try standard BraTS naming pattern with .nii.gz
    t1_path = os.path.join(case_path, f"{case_name}_t1.nii.gz")
    t1ce_path = os.path.join(case_path, f"{case_name}_t1ce.nii.gz")
    t2_path = os.path.join(case_path, f"{case_name}_t2.nii.gz")
    flair_path = os.path.join(case_path, f"{case_name}_flair.nii.gz")
    seg_path = os.path.join(case_path, f"{case_name}_seg.nii.gz")

    # If .nii.gz files don't exist, try .nii files
    if not os.path.exists(t1_path):
        t1_path = os.path.join(case_path, f"{case_name}_t1.nii")
    if not os.path.exists(t1ce_path):
        t1ce_path = os.path.join(case_path, f"{case_name}_t1ce.nii")
    if not os.path.exists(t2_path):
        t2_path = os.path.join(case_path, f"{case_name}_t2.nii")
    if not os.path.exists(flair_path):
        flair_path = os.path.join(case_path, f"{case_name}_flair.nii")
    if not os.path.exists(seg_path):
        seg_path = os.path.join(case_path, f"{case_name}_seg.nii")

    # Check which files exist
    paths = [t1_path, t1ce_path, t2_path, flair_path, seg_path]
    names = ["T1", "T1ce", "T2", "FLAIR", "seg"]

    existing_paths = {}
    for path, name in zip(paths, names):
        if os.path.exists(path):
            existing_paths[name] = path

    # If standard naming doesn't work, try alternative patterns
    if len(existing_paths) < 3:  # Need at least 3 files
        print(f"Standard naming not found for {case_name}. Available files:")
        for f in nii_files:
            print(f"  - {f}")

        # Try keyword-based matching for both .nii and .nii.gz files
        file_map = {}
        for f in nii_files:
            full_path = os.path.join(case_path, f)
            f_lower = f.lower()

            if '_t1.' in f and '_t1c' not in f_lower:
                file_map['T1'] = full_path
            elif '_t1c' in f_lower or 't1gd' in f_lower:
                file_map['T1ce'] = full_path
            elif '_t2.' in f and 'flair' not in f_lower:
                file_map['T2'] = full_path
            elif 'flair' in f_lower:
                file_map['FLAIR'] = full_path
            elif 'seg' in f_lower:
                file_map['seg'] = full_path

        existing_paths.update(file_map)

    return (
        existing_paths.get('T1'),
        existing_paths.get('T1ce'),
        existing_paths.get('T2'),
        existing_paths.get('FLAIR'),
        existing_paths.get('seg')
    )


def process_brats_case(base_dir, case_name, output_path):
    """Process a single BraTS case and save as h5 file"""

    try:
        # Find the case directory (in HGG or LGG)
        case_path = find_case_directory(base_dir, case_name)

        if case_path is None:
            print(f"❌ Case directory not found: {case_name}")
            return False

        # Determine if case is HGG or LGG
        case_type = "HGG" if "/HGG/" in case_path else "LGG" if "/LGG/" in case_path else "Unknown"

        # Find the image files
        t1_path, t1ce_path, t2_path, flair_path, seg_path = find_case_files(case_path, case_name)

        if seg_path is None:
            print(f"❌ No segmentation file found for {case_name}")
            return False

        # Choose primary modality (prefer T2, then FLAIR, then T1ce, then T1)
        image_path = None
        modality_used = ""

        for path, name in [(t2_path, "T2"), (flair_path, "FLAIR"), (t1ce_path, "T1ce"), (t1_path, "T1")]:
            if path and os.path.exists(path):
                image_path = path
                modality_used = name
                break

        if image_path is None:
            print(f"❌ No suitable image modality found for {case_name}")
            return False

        # Load image and segmentation
        image_nii = nib.load(image_path)
        seg_nii = nib.load(seg_path)

        # Get data
        image_data = image_nii.get_fdata()
        seg_data = seg_nii.get_fdata()

        # Check if data is valid
        if image_data.size == 0 or seg_data.size == 0:
            print(f"❌ Empty data for {case_name}")
            return False

        # Normalize image
        image_data = normalize_image(image_data)

        # Convert segmentation to binary (whole tumor vs background)
        # BraTS labels: 0=background, 1=necrotic, 2=edema, 4=enhancing
        label_data = (seg_data > 0).astype(np.uint8)

        # Resize to target shape
        try:
            from scipy.ndimage import zoom
        except ImportError:
            print("❌ scipy not available. Install with: pip install scipy")
            return False

        target_shape = (192, 192, 64)
        zoom_factors = [t / s for t, s in zip(target_shape, image_data.shape)]

        # Resize
        image_resized = zoom(image_data, zoom_factors, order=1)
        label_resized = zoom(label_data, zoom_factors, order=0)

        # Ensure correct types and shape
        image_resized = image_resized.astype(np.float32)
        label_resized = (label_resized > 0.5).astype(np.uint8)

        # Save as h5 file
        h5_filename = f"{case_name}.h5"
        h5_path = os.path.join(output_path, h5_filename)

        with h5py.File(h5_path, 'w') as h5f:
            h5f.create_dataset('image', data=image_resized, compression='gzip')
            h5f.create_dataset('label', data=label_resized, compression='gzip')
            h5f.attrs['modality'] = modality_used
            h5f.attrs['case_type'] = case_type
            h5f.attrs['original_shape'] = image_data.shape
            h5f.attrs['case_name'] = case_name

        print(f"✅ {case_name} ({case_type}): {image_data.shape} -> {target_shape} ({modality_used})")
        return True

    except Exception as e:
        print(f"❌ Error processing {case_name}: {str(e)}")
        return False


def get_all_available_cases(base_dir):
    """Get all available case names from HGG and LGG directories"""

    all_cases = []

    for subdir in ['HGG', 'LGG']:
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.exists(subdir_path):
            cases = [d for d in os.listdir(subdir_path)
                     if os.path.isdir(os.path.join(subdir_path, d)) and d.startswith('BraTS19')]
            all_cases.extend(cases)
            print(f"Found {len(cases)} cases in {subdir}")

    return sorted(all_cases)


def preprocess_brats2019(input_dir, output_dir, split_file_path):
    """Preprocess BraTS2019 dataset"""

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    data_output_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_output_dir, exist_ok=True)

    # Check split file
    if not os.path.exists(split_file_path):
        print(f"❌ Split file not found: {split_file_path}")
        print("💡 Available cases in dataset:")
        available_cases = get_all_available_cases(input_dir)
        for i, case in enumerate(available_cases[:10]):
            print(f"  - {case}")
        if len(available_cases) > 10:
            print(f"  ... and {len(available_cases) - 10} more cases")
        return

    # Read case names from split file
    with open(split_file_path, 'r') as f:
        case_names = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Split file contains {len(case_names)} cases")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {data_output_dir}")

    # Check what cases are actually available
    available_cases = get_all_available_cases(input_dir)
    print(f"Dataset contains {len(available_cases)} total cases")

    # Find intersection
    available_from_split = [case for case in case_names if case in available_cases]
    missing_cases = [case for case in case_names if case not in available_cases]

    print(f"Cases from split file available in dataset: {len(available_from_split)}")
    if missing_cases:
        print(f"Cases from split file NOT found in dataset: {len(missing_cases)}")
        print("First few missing cases:", missing_cases[:5])

    print("=" * 60)

    successful = 0
    failed = 0

    # Process only available cases
    for case_name in tqdm(available_from_split, desc="Processing cases"):
        if process_brats_case(input_dir, case_name, data_output_dir):
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 H5 files saved to: {data_output_dir}")

    if missing_cases:
        print(f"⚠️  {len(missing_cases)} cases from split file were not found in dataset")


if __name__ == "__main__":
    # Configuration
    RAW_DATA_DIR = "./MICCAI_BraTS_2019_Data_Training"
    OUTPUT_DIR = "../data/BraTS2019"

    # Process training data
    TRAIN_SPLIT_FILE = "../data/BraTS2019/train.txt"
    if not os.path.exists(TRAIN_SPLIT_FILE):
        TRAIN_SPLIT_FILE = "./data/BraTS2019/train.txt"

    print("🔄 Processing training data...")
    preprocess_brats2019(RAW_DATA_DIR, OUTPUT_DIR, TRAIN_SPLIT_FILE)

    # Process test data
    TEST_SPLIT_FILE = "../data/BraTS2019/test.txt"
    if not os.path.exists(TEST_SPLIT_FILE):
        TEST_SPLIT_FILE = "./data/BraTS2019/test.txt"

    print("\n🔄 Processing test data...")
    preprocess_brats2019(RAW_DATA_DIR, OUTPUT_DIR, TEST_SPLIT_FILE)

    print("\n🏁 Preprocessing completed!")