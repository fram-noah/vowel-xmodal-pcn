import pcn

class Params:
    framerate = 30
    audiorate = 44100


def default_params():
    """Set default parameters for PCN construction."""


def make_network(params):
    # Start with a simple model with a single hidden layer
    net = pcn.PCNetwork()



    with net:
        l_img_input = pcn.Layer(dim=image_dim, activation=pcn.Direct(), label="image")
        l_aud_input = pcn.AuditoryInput(
            n_samples=4096, sr=16000, n_fft=512, hop=256, n_mels=32,
            griffin_lim_iters=4, label="aud"
        )
        # Hidden layers
        l_img_hidden = pcn.Layer(dim=hidden_dim, activation=pcn.LeakyRelu(), label="img_h1")
        l_aud_hidden = pcn.Layer(dim=hidden_dim, activation=pcn.LeakyRelu(), label="aud_h1")
        # Joint output layer
        l_out = pcn.Layer(dim=n_classes, activation=pcn.LeakyRelu(), label="output")

        # Define edges connecting within layers