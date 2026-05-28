import torch

from particle import Particle


def get_ref_signal(ref_signal):  # Here we can define them all
    signal_decay = None
    if "inclusive" in ref_signal:
        signal_decay = {}
    elif 'BsToJpsiPhi' in ref_signal:
        signal_decay = {'final': torch.tensor([-13, 13, -321, 321]), 'head': [531, -531],
                        'LCA': torch.tensor([1, 1, 1, 1, 1, 1])}
    elif "BdToJpsiKst" in ref_signal:
        signal_decay = {'final': torch.tensor([-13, 13, -211, 321]), 'head': [511, -511],
                        'LCA': torch.tensor([1, 1, 1, 1, 1, 1])}
    elif "BuToJpsiK" in ref_signal:
        signal_decay = {'final': torch.tensor([-13, 13, 321]), 'head': [521, -521],
                        'LCA': torch.tensor([1, 1, 1])}
    elif "BdToDPi" in ref_signal: # cheated to dev one could put lca score to 1
        signal_decay = {'final': torch.tensor([211, 211, -321, -211]), 'head': [511, -511],
                        'LCA': torch.tensor([2, 2, 2, 1, 1, 1])}
    elif "BsToDsPi" in ref_signal:
        signal_decay = {'final': torch.tensor([211, 321, -321, -211]), 'head': [531, -531],
                        'LCA': torch.tensor([2, 2, 2, 1, 1, 1])}

    # Something to consider is that the topoLCA can differ, as they are dependent on gammactau
    # So intermediate states which could be part of both might cause struggles
    # This needs to be investigated
    if signal_decay is None:
        raise NotImplementedError("Topology of signal decay mode not defined yet, please add")
    return signal_decay
    # FT: Bs->DsPi, Bd->Dpi (BdToJpsiKs, BdToJpsipipi)
    # SL: BsToKmunu, BcToJpsitaunu BcToJpsimunu


def sig_matching(component, signal, mode='reco'):
    # Check if the number of particles are matching
    if len(signal['final']) != len(component['nodes']):
        return False

    # Matching pid of finals
    sel_pid = torch.sort(component['part_id']).values
    true_pos_pid, true_neg_pid = torch.sort(signal['final']).values, torch.sort(-1 * signal['final']).values
    if not torch.equal(sel_pid, true_pos_pid) and not torch.equal(sel_pid, true_neg_pid):
        return False

    # Matching LCA information on the edges
    if mode == 'reco':
        target_lca = torch.sort(torch.cat([signal['LCA'], signal['LCA']])).values
    elif mode == 'true':
        target_lca = torch.sort(signal['LCA']).values
    else:
        raise ValueError('mode must be reco or true')
    sel_lca = torch.sort(component['lca']).values
    if not torch.equal(sel_lca, target_lca):
        return False

    # Wrong head
    if mode == 'true':
        if not component['head_id'] in signal['head']:
            return False

    return True


def particle_name(id_):
    if id_ == 0:
        return 'ghost'
    elif id_ == 10413:
        return 'D1(2420)+'
    elif id_ == -10413:
        return 'D1(2420)-'
    elif id_ == 4412:
        return 'Sigma_cc+'
    elif id_ == -4412:
        return 'Sigma_cc-'
    elif id_ == 4422:
        return 'Chi_cc++'
    elif id_ == -4422:
        return 'Chi_cc--'
    elif id_ == 4432:
        return 'Omega_cc++'
    elif id_ == -4432:
        return 'Omega_cc--'
    else:
        try:
            name = Particle.from_pdgid(id_).name
        except:
            print(id_)
            name = str(id_)
        return name


pid_dict = {
    "Prob_k": 321,
    "Prob_e": 11,
    "Prob_mu": 13,
    "Prob_p": 2212,
    "Prob_pi": 211,
    "Prob_ghost": -1
}
