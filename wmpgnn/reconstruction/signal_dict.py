from particle import Particle


def get_ref_signal(ref_signal):  # Here we can define them all
    splitted = ref_signal.split("_")
    if splitted[-1] == "inclusive":
        return {}
    else:
        ref_signal = f"{splitted[-2]}_{splitted[-1]}"
    if 'Bs_Jpsiphi' == ref_signal:
        signal_decay = {'daughters': ['mu+', 'mu-', 'K+', 'K-'], 'mothers': ['B(s)0']}
        cc_signal_decay = {'daughters': ['mu+', 'mu-', 'K+', 'K-'], 'mothers': ['B(s)~0']}
        return signal_decay, cc_signal_decay
    elif "Bd_JpsiKst" == ref_signal:
        signal_decay = {'daughters': ['mu+', 'mu-', 'K+', 'pi-'], 'mothers': ['B0']}
        cc_signal_decay = {'daughters': ['mu+', 'mu-', 'pi+', 'K-'], 'mothers': ['B~0']}
        return signal_decay, cc_signal_decay
    elif 'Bd_JpsiKs' == ref_signal:
        signal_decay = {'daughters': ['mu+', 'mu-', 'pi+', 'pi-'], 'mothers': ['B0']}
        cc_signal_decay = {'daughters': ['mu+', 'mu-', 'pi+', 'pi-'], 'mothers': ['B~0']}
        return signal_decay, cc_signal_decay
    elif "Bs_Dspi" == ref_signal:
        signal_decay = {'daughters': ['K+', 'K-', 'pi+', 'pi-'], 'mothers': ['B(s)0']}
        cc_signal_decay = {'daughters': ['K+', 'K-', 'pi+', 'pi-'], 'mothers': ['B(s)~0']}
        return signal_decay, cc_signal_decay
    raise NotImplementedError


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
