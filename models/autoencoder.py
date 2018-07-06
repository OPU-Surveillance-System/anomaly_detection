import torch

def downsampling_block(in_dim, nb_f, nb_l):
    layers = []
    for n in range(nb_l):
        layers.append(torch.nn.Conv2d(in_dim, nb_f, (3, 3), padding=1))
        layers.append(torch.nn.ReLU())
        in_dim = nb_f
    layers.append(torch.nn.MaxPool2d((2, 2), (2, 2)))

    return layers

def upsampling_block(in_dim, nb_f, nb_l):
    layers = [torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
    for n in range(nb_l):
        layers.append(torch.nn.Conv2d(in_dim, nb_f, (3, 3), padding=1))
        layers.append(torch.nn.ReLU())
        in_dim = nb_f

    return layers

class Autoencoder(torch.nn.Module):
    def __init__(self, nb_f, nb_l, nb_b, fc):
        super(Autoencoder, self).__init__()

        self.nb_f = nb_f
        self.nb_l = nb_l
        self.nb_b = nb_b
        self.fc = fc

        #Encoder
        layers = []
        prev_in = 3
        prev_f = self.nb_f
        for n in range(self.nb_b):
            layers += downsampling_block(prev_in, prev_f, self.nb_l)
            prev_in = prev_f
            prev_f *= 2
        self.encoder = torch.nn.Sequential(*layers)

        #Bottleneck
        if self.fc:
            in_dim = ((224//(2**self.nb_b))**2)*(prev_f//2) #last_feature_map.h * last_feature_map.w * last_feature_map.c
            layers = [torch.nn.Linear(in_dim, self.fc), torch.nn.Linear(self.fc, in_dim)]
            self.bottleneck = torch.nn.Sequential(*layers)

        #Decoder
        layers = []
        for n in range(self.nb_b):
            prev_f //= 2
            layers += upsampling_block(prev_f, prev_f//2, self.nb_l)
        layers.append(torch.nn.Conv2d(prev_f//2, 3, (3, 3), padding=1))
        self.decoder = torch.nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.encoder(x)
        if self.fc:
            x = x.view(x.size(0), -1)
            x = self.bottleneck(x)
            reshape = 224//(2**self.nb_b)
            x = x.view(x.size(0), -1, reshape, reshape)
        logits = self.decoder(x)

        return logits

class VGGAutoencoder(torch.nn.Module):
    def __init__(self):
        super(VGGAutoencoder, self).__init__()

        def downsampling_block(in_dim, nb_f, nb_l):
            layers = []
            for n in range(nb_l):
                layers.append(torch.nn.Conv2d(in_dim, nb_f, (3, 3), padding=1))
                layers.append(torch.nn.ReLU())
                in_dim = nb_f
            layers.append(torch.nn.MaxPool2d((2, 2), (2, 2)))

            return layers

        def upsampling_block(in_dim, nb_f, nb_l):
            layers = [torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
            layers.append(torch.nn.ReLU())
            for n in range(nb_l):
                layers.append(torch.nn.Conv2d(in_dim, nb_f, (3, 3), padding=1))
                layers.append(torch.nn.ReLU())
                in_dim = nb_f

            return layers

        #Encoder
        layers = downsampling_block(3, 64, 2) + downsampling_block(64, 128, 2) + downsampling_block(128, 256, 3) + downsampling_block(256, 512, 3) + downsampling_block(512, 512, 3)
        self.encoder = torch.nn.Sequential(*layers)

        #Decoder
        layers = upsampling_block(512, 512, 3) + upsampling_block(512, 256, 3) + upsampling_block(256, 128, 3) + upsampling_block(128, 64, 2) + upsampling_block(64, 3, 2)
        self.decoder = torch.nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.encoder(x)
        logits = self.decoder(x)

        return logits
