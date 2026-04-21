import os

vqa_dir = './datasets/Kvasir-VQA/images'
seg_dir = './datasets/Kvasir-SEG/images'


#Compute hash for all images in both directories
def compute_hashes(directory):
    hashes = {}
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as f:
                file_hash = hash(f.read())
                hashes[file_hash] = filename
    return hashes


vqa_hashes = compute_hashes(vqa_dir)
seg_hashes = compute_hashes(seg_dir)

#Find common hashes
common_hashes = set(vqa_hashes.keys()) & set(seg_hashes.keys())
print(f'Number of common images: {len(common_hashes)}')