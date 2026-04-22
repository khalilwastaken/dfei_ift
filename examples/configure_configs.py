import glob
import yaml

def represent_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(list, represent_list, Dumper=yaml.SafeDumper)

if __name__ == "__main__":
    data_dir = "/eos/user/y/yukaiz/FT_Run3/data/DFEI_LHCb_LHCbcollision_normed_pt_data/"
    trn_sample = ["00342629_Bs_Jpsiphi", "00342442_inclusive", "00338627_Bd_JpsiKst"]
    trn_files = [1, 1, 1]

    eval_sample = ["00342629_Bs_Jpsiphi", "00342638_Bs_Jpsiphi"]
    eval_files = [1, 1]

    files = glob.glob("train*")
    for file in files:
        with open(file, "r") as f:
            config = yaml.safe_load(f)

        config["settings"]["data_dir"] = data_dir
        config["settings"]["sample"] = trn_sample
        config["settings"]["nfiles"] = trn_files
        config["evaluate"]["sample"] = eval_sample
        config["evaluate"]["nfiles"] = eval_files

        with open(file, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)