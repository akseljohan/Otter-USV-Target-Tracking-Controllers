import casadi as ca
from trash import casadiMPC_dev as campc
import numpy as np

#Test inputs

I = np.identity(3)

tmCRB_v = ca.DM([[0,-1,1],[1,0,0],[1,0,0]])

tmCA_vr = ca.DM[[0,-1,1],[1,0,0],[1,0,0]]


# content of test_class.py
class TestCasadiMPC:
    def test_CRB_v(self):
        print(tmCRB_v)
        assert ca.is_equal(campc.CRB_v(1,1,1) , tmCRB_v)

    def test_CA_vr(self):
        assert ca.is_equal(campc.CA_vr(), )