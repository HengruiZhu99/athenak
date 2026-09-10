import unittest
from pathlib import Path
from controller import campaign_input

def parameters(text):
    result={};section=None
    for raw in text.splitlines():
        line=raw.split('#',1)[0].strip()
        if line.startswith('<') and line.endswith('>'):section=line[1:-1]
        elif '=' in line:
            k,v=line.split('=',1); key=(section,k.strip())
            if key in result:raise ValueError('duplicate parameter '+str(key))
            result[key]=v.strip()
    return result

class CampaignInput(unittest.TestCase):
    def test_only_authorized_parameters_change(self):
        baseline=Path(__file__).with_name('baseline.athinput').read_text()
        old=parameters(baseline);new=parameters(campaign_input(baseline,Path('/isolated/cycle_01')))
        allowed={('job','basename'),('time','tlim'),('fastflow','num_horizons'),
            ('problem','stop_on_horizon'),('problem','stop_on_dispersion'),('problem','collapse_lapse_threshold'),('problem','dispersion_lapse_dip'),('problem','dispersion_lapse_recovery'),
            ('mesh_refinement','amr_history_file'),('problem','brill_global_coefficients_file'),
            ('problem','constraint_summary_file'),('z4c','boundary_rhs'),('z4c','extrap_order'),
            ('z4c','vc_single_rank_device_sync'),('z4c','history_constraint_radius'),('output5','dt')}
        changes={k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
        self.assertLessEqual(changes,allowed)
        self.assertEqual(new['mesh_refinement','amr_history_mode'],'record')
        self.assertEqual(new['z4c','boundary_rhs'],'full_constraint_bjorhus')
        self.assertEqual(new['z4c','extrap_order'],'2')
        self.assertEqual(new['z4c','vc_single_rank_device_sync'],'true')
        self.assertEqual(new['time','tlim'],'200')
        self.assertEqual(float(new['problem','collapse_lapse_threshold']),1e-5)
        self.assertEqual(new['fastflow','num_horizons'],'0')
        self.assertEqual(new['problem','stop_on_horizon'],'false')
        self.assertEqual(new['problem','stop_on_dispersion'],'false')

    def test_cases_differ_only_in_history_path(self):
        baseline=Path(__file__).with_name('baseline.athinput').read_text()
        a=parameters(campaign_input(baseline,Path('/isolated/a')))
        b=parameters(campaign_input(baseline,Path('/isolated/b')))
        self.assertEqual({k for k in a if a[k]!=b[k]},{('mesh_refinement','amr_history_file')})

if __name__=='__main__':unittest.main()
