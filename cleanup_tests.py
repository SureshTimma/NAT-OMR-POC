import os
import glob

def cleanup_test_dir(test_dir):
    print(f"Cleaning {test_dir}...")
    
    # Remove raw section crops
    for f in glob.glob(os.path.join(test_dir, "debug_section_*.jpg")):
        try:
            os.remove(f)
            print(f"  Deleted {os.path.basename(f)}")
        except Exception as e:
            print(f"  Failed delete {f}: {e}")

    # Remove old result name
    old_result = os.path.join(test_dir, "result_evaluated.jpg")
    if os.path.exists(old_result):
        try:
            os.remove(old_result)
            print(f"  Deleted result_evaluated.jpg")
        except Exception as e:
            print(f"  Failed delete {old_result}: {e}")

    # Remove debug_edges.jpg
    edges_file = os.path.join(test_dir, "debug_edges.jpg")
    if os.path.exists(edges_file):
        try:
            os.remove(edges_file)
            print(f"  Deleted debug_edges.jpg")
        except Exception as e:
            print(f"  Failed delete {edges_file}: {e}")

    # Remove debug_sections.jpg
    sections_file = os.path.join(test_dir, "debug_sections.jpg")
    if os.path.exists(sections_file):
        try:
            os.remove(sections_file)
            print(f"  Deleted debug_sections.jpg")
        except Exception as e:
            print(f"  Failed delete {sections_file}: {e}")

    # Remove threshold debug images
    for f in glob.glob(os.path.join(test_dir, "*_thresh*.jpg")):
        try:
            os.remove(f)
            print(f"  Deleted {os.path.basename(f)}")
        except Exception as e:
            print(f"  Failed delete {f}: {e}")

    # Remove individual section debug images
    for f in glob.glob(os.path.join(test_dir, "debug_section_*_bubbles.jpg")):
        try:
            os.remove(f)
            print(f"  Deleted {os.path.basename(f)}")
        except Exception as e:
            print(f"  Failed delete {f}: {e}")

def main():
    base_test_dir = os.path.join(os.getcwd(), "test")
    if not os.path.exists(base_test_dir):
        print("No test directory found.")
        return

    # cleanup root directory too if ran there previously
    cleanup_test_dir(os.getcwd())
    
    # cleanup subdirectories
    subdirs = [d for d in os.listdir(base_test_dir) if os.path.isdir(os.path.join(base_test_dir, d))]
    for d in subdirs:
        cleanup_test_dir(os.path.join(base_test_dir, d))

if __name__ == "__main__":
    main()
