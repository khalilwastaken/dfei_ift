def load_trn_val_loader(configs, model="DFEI"):
    print("Obtaining train and validation loaders:")
    trn_loader, val_loader, tst_loader, chunkloader = None, None, None, None
    if "true" in configs[model]["settings"]["graph_mode"]:
        from wmpgnn.data_loader.data_loader import get_trn_val_loaders

        trn_loader, val_loader, weights, nevts = get_trn_val_loaders(configs[model])
        configs[model].update({"num_events": nevts})
        return configs, weights, trn_loader, val_loader, chunkloader

    # here full event using chunkloading everything normal loading -> pv asso should also end here
    if "nu7p6" in configs[model]["settings"]["data_dir"] or "LHCbcollision" in configs[model]["settings"]["data_dir"]:
        from wmpgnn.data_loader.chunk_loader import get_trn_val_loaders

        chunkloader = get_trn_val_loaders(configs[model])
        print("Obtaining weights:")
        weights = chunkloader.trn_dataset.get_weights()
        configs[model].update({"num_files": chunkloader.trn_dataset.n_files})
    else:
        from wmpgnn.data_loader.data_loader import get_trn_val_loaders

        trn_loader, val_loader, weights, nevts = get_trn_val_loaders(configs[model])
        configs[model].update({"num_events": nevts})
    return configs, weights, trn_loader, val_loader, chunkloader




def load_tst_loader(configs, model="DFEI"):
    print("Obtaining test loaders:")
    tst_loader, chunkloader = None, None
    if "true" in configs[model]["settings"]["graph_mode"]:
        from wmpgnn.data_loader.data_loader import get_tst_loader

        tst_loader, nevts = get_tst_loader(configs, model=model)
        configs[model].update({"num_events": nevts})
        return configs, tst_loader, chunkloader

    if "nu7p6" in configs[model]["settings"]["data_dir"]:
        from wmpgnn.data_loader.chunk_loader import get_tst_loader

        chunkloader = get_tst_loader(configs, model=model)
        configs[model].update({"num_files": chunkloader.tst_dataset.n_files})
    else:
        from wmpgnn.data_loader.data_loader import get_tst_loader

        tst_loader, nevts = get_tst_loader(configs, model=model)
        configs[model].update({"num_events": nevts})

    return configs, tst_loader, chunkloader
