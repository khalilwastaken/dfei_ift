import argparse
import sys, os
import yaml
import glob
from tqdm import tqdm
import shutil

from multiprocessing.pool import ThreadPool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from wmpgnn.pv_association.pv_asso_lm import obtain_pv_model
from wmpgnn.analysis.config_adjusting import determine_ncpus
from wmpgnn.data_loader.data_loader_class import DataSetLoader
from wmpgnn.data_loader.helper import save_compressed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Config file path")
    args = parser.parse_args()
    print("=" * 45)
    print("Starting the pv association to disk script")
    # Load config file
    with open(args.config, "r") as file:
        configs = determine_ncpus(yaml.safe_load(file))
        configs['log_dir'] = 'LHCb_logs'

    indir = configs["settings"]["data_dir"]
    outdir = f"{os.path.dirname(indir.rstrip('/'))}/PV_asso_DFEI_v{configs['settings']['pv_model']}_{os.path.basename(indir.rstrip('/'))}"
    for sample in  configs['settings']['sample']:
        os.makedirs(f"{outdir}/{sample}", exist_ok=True)
        yaml_file = glob.glob(f'{indir}/{sample}/*.yaml')[0]
        shutil.copy(yaml_file, f"{outdir}/{sample}/")

    # Get the PV model
    pv_model = obtain_pv_model(configs)
    # Get the data loader class
    dataloader = DataSetLoader(configs, pv_model)

    files = sorted(
        f
        for sample in configs['settings']['sample']
        for f in glob.glob(f'{indir}/{sample}/*.pt.zst')
    )


    def process_file(_file):
        name = os.path.basename(_file)
        sample = os.path.basename(os.path.dirname(_file))
        data = dataloader.load_data(_file)
        save_compressed(data, f"{outdir}/{sample}/{name}")


    with ThreadPool(processes=configs["ncpus"]["loading"]) as pool:
        list(tqdm(pool.imap(process_file, files), total=len(files),
                  desc=f"Loading {configs['settings']['sample']} training dataset"))
    print("Done")
    print("=" * 45)
