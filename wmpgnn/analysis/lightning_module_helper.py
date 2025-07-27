def init_logs(configs):
    trn_log = {}
    if configs["LCA"]:
        trn_log["LCA_loss"] = []
        for i in range(4):  # something like num classes in config file
            trn_log[f"LCA_class{i}_num"] = []
            for j in range(4):
                trn_log[f"LCA_class{i}_pred_class{j}"] = []

    return trn_log, trn_log