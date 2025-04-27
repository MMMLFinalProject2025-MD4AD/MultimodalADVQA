import os
import argparse
from nuscenes.nuscenes import NuScenes

def get_image_paths_for_sample_token(nusc, sample_token):
    """
    Given a sample_token, fetch and return the paths to all images for that sample.
    """
    try:
        sample = nusc.get('sample', sample_token)
    except KeyError:
        print(f"Error: Sample token {sample_token} not found in the dataset.")
        return []

    # Check if 'data' exists in the sample
    if 'data' not in sample:
        print(f"No sensor data found for sample_token {sample_token}.")
        return []

    # Initialize the list to store image paths
    image_paths = []

    # Iterate over the cameras in the sample data
    for cam_token in sample['data'].values():
        try:
            cam = nusc.get('sample_data', cam_token)  # Get the sensor data
            if cam['is_key_frame']:  # Only consider keyframe images
                image_path = os.path.join(nusc.dataroot, cam['filename'])
                image_paths.append(image_path)
        except KeyError:
            print(f"Error: sample_data token {cam_token} not found.")
            continue

    return image_paths    

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_token", required=True, type=str, help="The sample_token to query")
    parser.add_argument("--output_file", required=True, type=str, help="Output file to save the image paths")
    parser.add_argument("--dataroot", required=True, type=str, help="Path to the nuScenes dataset")    
    args = parser.parse_args()

    # Initialize the NuScenes object
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=True)

    # Get the image paths for the specified sample_token
    image_paths = get_image_paths_for_sample_token(nusc, args.sample_token)
    
    # Write the image paths to the output file
    with open(args.output_file, 'w') as f:
        if image_paths:
            for path in image_paths:
                f.write(path + "\n")
            print(f"Image paths for sample_token {args.sample_token} written to {args.output_file}")
        else:
            f.write(f"No images found for sample_token {args.sample_token}.\n")
            print(f"No images found for sample_token {args.sample_token}.")

if __name__ == "__main__":
    main()