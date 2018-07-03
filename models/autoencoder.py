import torch

class Autoencoder(torch.nn.Module):
    def __init__(self, nb_f, nb_l, nb_b, fc):
        super(Autoencoder, self).__init__()

        self.nb_f = nb_f
        self.nb_l = nb_l
        self.nb_b = nb_b
        self.fc = fc

        def downsampling_block(in_dim, nb_f, nb_l):
            layers = []
            for n in range(nb_l):
                layers.append(torch.nn.Conv2d(in_dim, nb_f, (3, 3), padding=1))
                layers.append(torch.nn.ReLU())
                in_dim = nb_f
            layers.append(torch.nn.MaxPool2d((2, 2), (2, 2)))

            return layers

        def upsampling_block(in_dim, nb_f, nb_l):
            layers = [torch.nn.Upsample(scale_factor=2, mode='bilinear')]
            layers.append(torch.nn.ReLU())
            for n in range(nb_l):
                layers.append(torch.nn.Conv2d(in_dim, nb_f, (3, 3), padding=1))
                layers.append(torch.nn.ReLU())
                in_dim = nb_f

            return layers

        #Encoder
        layers = []
        prev_in = 1
        prev_f = self.nb_f
        for n in range(self.nb_b):
            layers += downsampling_block(prev_in, prev_f, self.nb_l)
            prev_in = prev_f
            prev_f *= 2
        self.encoder = torch.nn.Sequential(*layers)

        #Bottleneck
        if self.fc:
            in_dim = ((28//(2**self.nb_b))**2)*(prev_f//2) #last_feature_map.h * last_feature_map.w * last_feature_map.c
            layers = [torch.nn.Linear(in_dim, self.fc), torch.nn.Linear(self.fc, in_dim)]
            self.bottleneck = torch.nn.Sequential(*layers)

        #Decoder
        layers = []
        for n in range(self.nb_b):
            prev_f //= 2
            layers += upsampling_block(prev_f, prev_f//2, self.nb_l)
        layers.append(torch.nn.Conv2d(prev_f//2, 1, (3, 3), padding=1))
        self.decoder = torch.nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal(m.weight)

    def forward(self, x):
        x = self.encoder(x)
        if self.fc:
            x = x.view(x.size(0), -1)
            x = self.bottleneck(x)
            reshape = 28//(2**self.nb_b)
            x = x.view(x.size(0), -1, reshape, reshape)
        logits = self.decoder(x)

        return logits
