import os
import argparse
import sys
import subprocess
import glob
import shutil
import dvc.api

from collections import Counter
from git.repo.base import Repo
from smexperiments.tracker import Tracker
# from torchvision.datasets.utils import extract_archive
from utils import extract_archive
from sklearn.model_selection import train_test_split
from pathlib import Path


dvc_repo_url = os.environ.get('DVC_REPO_URL')
dvc_branch = os.environ.get('DVC_BRANCH')

git_user = os.environ.get('GIT_USER', "sagemaker")
git_email = os.environ.get('GIT_EMAIL', "sagemaker-processing@example.com")

ml_root = Path("/opt/ml/processing")

dataset_zip = ml_root / "input" / "flowers.zip"
git_path = ml_root / "sagemaker-flower"

def configure_git():
    subprocess.check_call(['git', 'config', '--global', 'user.email', f'"{git_email}"'])
    subprocess.check_call(['git', 'config', '--global', 'user.name', f'"{git_user}"'])
    
def clone_dvc_git_repo():
    print(f"\t:: Cloning repo: {dvc_repo_url}")
    
    repo = Repo.clone_from(dvc_repo_url, git_path.absolute())
    
    return repo

def sync_data_with_dvc(repo):
    os.chdir(git_path)
    print(f":: Create branch {dvc_branch}")
    try:
        repo.git.checkout('-b', dvc_branch)
        print(f"\t:: Create a new branch: {dvc_branch}")
    except:
        repo.git.checkout(dvc_branch)
        print(f"\t:: Checkout existing branch: {dvc_branch}")
    print(":: Add files to DVC")
    
    subprocess.check_call(['dvc', 'add', "dataset"])
    
    repo.git.add(all=True)
    repo.git.commit('-m', f"'add data for {dvc_branch}'")
    
    print("\t:: Push data to DVC")
    subprocess.check_call(['dvc', 'push'])
    
    print("\t:: Push dvc metadata to git")
    repo.remote(name='origin')
    repo.git.push('--set-upstream', repo.remote().name, dvc_branch, '--force')

    sha = repo.head.commit.hexsha
    
    print(f":: Commit Hash: {sha}")

def write_dataset(image_paths, output_dir):
    for img_path in image_paths:
        Path(output_dir / img_path.parent.stem).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(img_path, output_dir / img_path.parent.stem / img_path.name)

def generate_train_test_split():
    dataset_extracted = ml_root / "tmp"
    dataset_extracted.mkdir(parents=True, exist_ok=True)
    
    # split dataset and save to their directories
    print(f":: Extracting Zip {dataset_zip} to {dataset_extracted}")
    extract_archive(
        from_path=dataset_zip,
        to_path=dataset_extracted
    )
    
    dataset_full = list((dataset_extracted / "flowers").glob("*/*.jpg"))
    labels = [x.parent.stem for x in dataset_full]
    
    print(":: Dataset Class Counts: ", Counter(labels))
    
    d_train, d_test = train_test_split(dataset_full, stratify=labels)
    
    print("\t:: Train Dataset Class Counts: ", Counter(x.parent.stem for x in d_train))
    print("\t:: Test Dataset Class Counts: ", Counter(x.parent.stem for x in d_test))
    
    for path in ['train', 'test']:
        output_dir = git_path / "dataset" / path
        print(f"\t:: Creating Directory {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    print(":: Writing Datasets")
    write_dataset(d_train, git_path / "dataset" / "train")
    write_dataset(d_test, git_path / "dataset" / "test")
        
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    # setup git
    print(":: Configuring Git")
    configure_git()
    
    print(":: Cloning Git")
    repo = clone_dvc_git_repo()
    
    print(":: Generate Train Test Split")
    # extract the input zip file and split into train and test
    generate_train_test_split()
        
    print(":: copy data to train")
    subprocess.check_call('cp -r /opt/ml/processing/sagemaker-flower/dataset/train/* /opt/ml/processing/dataset/train', shell=True)
    subprocess.check_call('cp -r /opt/ml/processing/sagemaker-flower/dataset/test/* /opt/ml/processing/dataset/test', shell=True)
    
    print(":: Sync Processed Data to Git & DVC")
    sync_data_with_dvc(repo)

 