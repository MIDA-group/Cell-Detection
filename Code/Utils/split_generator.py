import os
import random
import shutil

def create_non_overlapping_splits(split1_path, num_splits=3):
    """Create splits with unique test sets."""
    # Directories for different splits
    dirs = {
        'train': ('train-images', 'train-labels'),
        'val': ('val-images', 'val-labels'),
        'test': ('test-images', 'test-labels')
    }

    # Collect valid pairs with precise tracking
    def get_valid_pairs(img_path, label_path):
        img_files = os.listdir(img_path)
        label_files = os.listdir(label_path)
        
        pairs = []
        for img in img_files:
            base_name = os.path.splitext(img)[0]
            possible_labels = [
                base_name + '.json', 
                base_name + '.txt',
                base_name.replace(' ', '_') + '.json',
                base_name.replace(' ', '_') + '.txt'
            ]
            
            matching_label = next((label for label in label_files if label in possible_labels), None)
            if matching_label:
                pairs.append(img)
        
        return pairs

    # Collect pairs for each split type
    split_pairs = {}
    split_counts = {}
    for split_type, (img_dir, label_dir) in dirs.items():
        img_path = os.path.join(split1_path, img_dir)
        label_path = os.path.join(split1_path, label_dir)
        split_pairs[split_type] = get_valid_pairs(img_path, label_path)
        split_counts[split_type] = len(split_pairs[split_type])
        print(f"{split_type} count: {split_counts[split_type]}")

    # Separate test pairs to ensure uniqueness
    test_pairs = split_pairs['test']
    train_val_pairs = split_pairs['train'] + split_pairs['val']
    random.shuffle(train_val_pairs)

    # Create additional splits
    base_dataset_dir = os.path.dirname(split1_path)
    used_test_pairs = set()

    for i in range(2, num_splits + 2):
        new_split_path = os.path.join(base_dataset_dir, f'split{i}')
        os.makedirs(new_split_path, exist_ok=True)

        # Select unique test pairs
        available_test = [p for p in test_pairs if p not in used_test_pairs]
        if len(available_test) < split_counts['test']:
            available_test.extend([p for p in test_pairs if p in used_test_pairs])
        
        new_test_pairs = available_test[:split_counts['test']]
        used_test_pairs.update(new_test_pairs)

        # Shuffle remaining train/val pairs
        random.shuffle(train_val_pairs)

        # Create new splits
        new_splits = {
            'test': new_test_pairs,
            'train': train_val_pairs[:split_counts['train']],
            'val': train_val_pairs[split_counts['train']:]
        }

        # Copy files for each split type
        for split_type, pairs in new_splits.items():
            img_src_dir = os.path.join(split1_path, dirs[split_type][0])
            label_src_dir = os.path.join(split1_path, dirs[split_type][1])
            
            img_dst_dir = os.path.join(new_split_path, dirs[split_type][0])
            label_dst_dir = os.path.join(new_split_path, dirs[split_type][1])
            
            os.makedirs(img_dst_dir, exist_ok=True)
            os.makedirs(label_dst_dir, exist_ok=True)

            for img in pairs:
                # Determine label filename
                base_name = os.path.splitext(img)[0]
                possible_labels = [
                    base_name + '.json', 
                    base_name + '.txt',
                    base_name.replace(' ', '_') + '.json',
                    base_name.replace(' ', '_') + '.txt'
                ]
                
                # Find first existing label
                label = next((l for l in possible_labels if os.path.exists(os.path.join(label_src_dir, l))), None)
                
                if label:
                    # Copy image
                    shutil.copy2(
                        os.path.join(img_src_dir, img), 
                        os.path.join(img_dst_dir, img)
                    )
                    
                    # Copy label
                    shutil.copy2(
                        os.path.join(label_src_dir, label), 
                        os.path.join(label_dst_dir, os.path.basename(label))
                    )

        # Verify counts
        for split_type in ['train', 'val', 'test']:
            dst_img_dir = os.path.join(new_split_path, dirs[split_type][0])
            print(f"Split{i} {split_type} count: {len(os.listdir(dst_img_dir))}")

        print(f"Created split{i}")

# Usage
# create_non_overlapping_splits('/path/to/split1')

split1_path = '/work/marco/SCIA2025/CNSeg/PatchSeg/split1'
create_non_overlapping_splits(split1_path)