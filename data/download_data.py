#import library
import sys #for input to running script
import zipfile #for processing .zip file
import subprocess #for running external command
from pathlib import Path #for made a path in a directory

#made a absolute path for directory raw for a destination directory
DATA_DIR = Path(__file__).resolve().parent / "raw"

#made a function ensure_data_dir to make sure that directory raw is already created
def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

#made a function to export dataset from kaggle using path website
def export_dataset(dataset: str = "muratkokludataset/rice-image-dataset"):
    ensure_data_dir() #call function ensure_data_dir() to access path directory raw
    cwd = DATA_DIR #made a variable as absolute path
    cmd = [
        "kaggle", "datasets", "download", #path kaggle to download
        "-d", dataset, #path dataset kaggle
        "-p", str(cwd), #path destination download dataset
        "--force" #force overwrite if the file exciting 
        ] 
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd) #run cmd "kaggle datasets download "

#made a function to unzip all files in directory raw
def unzip_all(delete_zip: bool = True): #if delete_zip = True, zip file will deleted after extraction
    """Unzip semua file .zip di folder data/raw. 
    Jika delete_zip=True, file zip akan dihapus setelah ekstraksi."""
    for z in DATA_DIR.glob("*.zip"):
        print("Unzipping", z)
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(DATA_DIR)
        if delete_zip:
            print("Deleting", z)
            z.unlink()


if __name__ == "__main__": #this code will running if this file directly executed
    dataset = sys.argv[1] if len(sys.argv) > 1 else "muratkokludataset/rice-image-dataset" #checking argument if not using additional argument so using default dataset "muratkokludataset/rice-image-dataset"
    export_dataset(dataset) #call function export_dataset to download dataset
    unzip_all(delete_zip=True)#call function unzip_all to unzip file .zip and deleted .zip file after extracted
    print("✅ Dataset Downloaded & Extracted to:", DATA_DIR)
