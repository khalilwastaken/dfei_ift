def load_trn_val_loader(configs):
    print("Obtaining train and validation loaders:")
    trn_loader, val_loader, tst_loader, chunkloader = None, None, None, None

    # PV association applied to the graphs
    if configs["settings"]["pv_model"] != "None":
        print("Using pv associated data loader")
        from wmpgnn.data_loader.pv_assoed_loader import get_trn_val_loaders

        trn_loader, val_loader, weights, nevts = get_trn_val_loaders(configs)
        configs.update({"num_events": nevts})
        return configs, weights, trn_loader, val_loader, chunkloader

    # If we do some prior filtering refer back to default loader
    if "true" in configs["settings"]["graph_mode"]:
        print("Using default data loader due to initial pruning")
        from wmpgnn.data_loader.default_data_loader import get_trn_val_loaders

        trn_loader, val_loader, weights, nevts = get_trn_val_loaders(configs)
        configs.update({"num_events": nevts})
        return configs, weights, trn_loader, val_loader, chunkloader

    # here full event using chunkloading
    data_dir = configs["settings"]["data_dir"]
    if "nu7p6" in data_dir or "LHCbcollision" in data_dir:
        print("Using chunk loader")
        from wmpgnn.data_loader.chunk_loader import get_trn_val_loaders

        chunkloader = get_trn_val_loaders(configs)
        print("Obtaining weights:")
        weights = chunkloader.trn_dataset.get_weights()
        configs.update({"num_files": chunkloader.trn_dataset.n_files})

        return configs, weights, trn_loader, val_loader, chunkloader

    # Default option
    print("Using default data loader")
    from wmpgnn.data_loader.default_data_loader import get_trn_val_loaders

    trn_loader, val_loader, weights, nevts = get_trn_val_loaders(configs)
    configs.update({"num_events": nevts})
    return configs, weights, trn_loader, val_loader, chunkloader


def load_tst_loader(configs):
    print("=" * 15)
    print("Obtaining test loaders:")
    tst_loader, chunkloader = None, None

    if configs["settings"]["pv_model"] != "None":
        print("Using pv associated data loader")
        from wmpgnn.data_loader.pv_assoed_loader import get_tst_loader

        tst_loader, nevts = get_tst_loader(configs)
        configs.update({"num_events": nevts})
        return configs, tst_loader, chunkloader

    if "true" in configs["settings"]["graph_mode"]:
        print("Using filtered data loader")
        from wmpgnn.data_loader.default_data_loader import get_tst_loader

        tst_loader, nevts = get_tst_loader(configs)
        configs.update({"num_events": nevts})
        return configs, tst_loader, chunkloader

    data_dir = configs["settings"]["data_dir"]
    if "nu7p6" in data_dir or "LHCbcollision" in data_dir or "LHCb_data" in data_dir:
        print("Using chunk loader")
        from wmpgnn.data_loader.chunk_loader import get_tst_loader

        chunkloader = get_tst_loader(configs)
        configs.update({"num_files": chunkloader.tst_dataset.n_files})
    else:
        print("Using data loader")
        from wmpgnn.data_loader.default_data_loader import get_tst_loader

        tst_loader, nevts = get_tst_loader(configs)
        configs.update({"num_events": nevts})

    return configs, tst_loader, chunkloader
