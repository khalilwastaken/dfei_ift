import torch

from particle import Particle


def get_ref_signal(ref_signal):  # Here we can define them all
    if ref_signal == "inclusive":
        return {}
    if 'BsToJpsiPhi' in ref_signal:
        signal_decay = {'final': torch.tensor([-13, 13, -321, 321]), 'head': [531, -531],
                        'LCA': torch.tensor([1, 1, 1, 1, 1, 1])}
        return signal_decay
    elif ref_signal in 'BdToJpsiKst':
        signal_decay = {'daughters': ['mu+', 'mu-', 'K+', 'pi-'], 'mothers': ['B0']}
        cc_signal_decay = {'daughters': ['mu+', 'mu-', 'pi+', 'K-'], 'mothers': ['B~0']}
        return signal_decay, cc_signal_decay
    elif ref_signal in 'BdToJpsiKs':
        signal_decay = {'daughters': ['mu+', 'mu-', 'pi+', 'pi-'], 'mothers': ['B0']}
        cc_signal_decay = {'daughters': ['mu+', 'mu-', 'pi+', 'pi-'], 'mothers': ['B~0']}
        return signal_decay, cc_signal_decay
    elif ref_signal in 'BsToDspi':
        signal_decay = {'daughters': ['K+', 'K-', 'pi+', 'pi-'], 'mothers': ['B(s)0']}
        cc_signal_decay = {'daughters': ['K+', 'K-', 'pi+', 'pi-'], 'mothers': ['B(s)~0']}
        return signal_decay, cc_signal_decay
    elif ref_signal in 'BsToKmunu':
        signal_decay = {'daughters': ['K-', 'mu+'], 'mothers': ['B(s)0']}
        cc_signal_decay = {'daughters': ['K+', 'mu-'], 'mothers': ['B(s)~0']}
        return signal_decay, cc_signal_decay
    elif ref_signal in 'BuToJpsiK':
        signal_decay = {'daughters': ['mu+', 'mu-', 'K+'], 'mothers': ['B+']}
        cc_signal_decay = {'daughters': ['mu+', 'mu-', 'K-'], 'mothers': ['B-']}
        return signal_decay, cc_signal_decay
    elif ref_signal in 'BcToJpsitaunu' or ref_signal in 'BcToJpsimunu':
        signal_decay = {'daughters': ['mu+', 'mu-', 'mu+'], 'mothers': ['B(c)+']}
        cc_signal_decay = {'daughters': ['mu+', 'mu-', 'mu-'], 'mothers': ['B(c)-']}
        return signal_decay, cc_signal_decay
    import pdb; pdb.set_trace()

    raise NotImplementedError


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
        target_lca =  torch.sort(torch.cat([signal['LCA'], signal['LCA']])).values
    elif mode == 'true':
        target_lca =  torch.sort(signal['LCA']).values
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
