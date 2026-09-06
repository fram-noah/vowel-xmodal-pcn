import pcn

class PCNDef:
    img_size = 1280
    aud_size = 4096
    n_img_hidden = 1
    n_aud_hidden = 1
    img_hidden_size = 512
    aud_hidden_size = 512
    n_joint_hidden = 0
    joint_hidden_size = 256
    n_classes = 2
    model_type = 'desc'
    framerate = 30
    audiorate = 44100
    audiomethod = 'pcn'
    n_fft = 512
    hop = 256
    mels = 32
    griffin_lim_iters = 4

def default_params():
    """Set default parameters for PCN construction."""


def make_network_sequential(pcndef):
    # Flexible for number of hidden and joint layers
    net = pcn.PCNetwork()
    with net:
        l_img_input = pcn.Layer(
            dim=pcndef.image_size,
            activation=pcn.Direct(),
            label="image"
        )
        l_aud_input = pcn.AuditoryInput(
            n_samples=pcndef.audio_size,
            sr=pcndef.audiorate,
            n_fft=pcndef.n_fft,
            hop=pcndef.hop,
            n_mels=pcndef.mels,
            griffin_lim_iters=pcndef.griffin_lim_iters,
            label="aud"
        )
        # Hidden layers
        l_img_hidden = [pcn.Layer(
            dim=pcndef.img_hidden_size,
            activation=pcn.LeakyRelu(),
            label="img_h%d" % i) for i in range(pcndef.n_img_hidden)]
        l_aud_hidden = [pcn.Layer(
            dim=pcndef.aud_hidden_size,
            activation=pcn.LeakyRelu(),
            label="aud_h%d" % i) for i in range(pcndef.n_aud_hidden)]
        # Joint output layer
        l_out = pcn.Layer(
            dim=pcndef.n_classes,
            activation=pcn.LeakyRelu(),
            label="output")
        # Joint layers (if present)
        if pcndef.n_joint_hidden > 0:
            l_joint_hidden = [pcn.Layer(
                dim=pcndef.joint_hidden_size,
                activation=pcn.LeakyRelu(),
                label="joint_h%d" % i) for i in range(pcndef.n_joint_hidden)]
            joint_layerlist = l_joint_hidden + [l_out]
        else:
            joint_layerlist = [l_out]
        # Define edges connecting within layers
        # For now, restricted to bottom-up sequential connections
        img_layerlist = [l_img_input] + l_img_hidden
        aud_layerlist = [l_aud_input] + l_aud_hidden
        map(lambda i: pcn.Predict(img_layerlist[i], img_layerlist[i+1]), range(len(img_layerlist) - 1))
        map(lambda i: pcn.Predict(aud_layerlist[i], aud_layerlist[i+1]), range(len(aud_layerlist) - 1))
        # Link sensory
        pcn.Predict([img_layerlist[-1], aud_layerlist[-1]], joint_layerlist[0])
        # Iterate through joint layers, if present
        if len(joint_layerlist) > 1:
            map(lambda i: pcn.Predict(joint_layerlist[i], joint_layerlist[i+1]), range(len(joint_layerlist) - 1))

    handles = dict(
        l_img_input=l_img_input,
        l_aud_input=l_aud_input,
        l_output=l_out
    )
    return net, handles