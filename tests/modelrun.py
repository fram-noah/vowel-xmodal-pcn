import pcn
from pcn_model import *
import matplotlib.pyplot as plt

pcndef = PCNDef(
    n_img_hidden=2,
    n_aud_hidden=3,
    n_joint_hidden=1
)

net, handles = make_network_sequential(pcndef)