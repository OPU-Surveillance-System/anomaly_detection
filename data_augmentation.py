import imgaug as ia
from imgaug import augmenters as iaa

def augment_batch(inputs):
    seq = iaa.Sequential(iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        #iaa.Invert(0.5),
        iaa.OneOf([
            iaa.Add((-5, 25)),
            iaa.Multiply((0.25, 1.5))
        ]),
        iaa.OneOf([
            iaa.OneOf([
                iaa.GaussianBlur((0, 1.0)),
                iaa.AverageBlur(k=(1, 5)),
                iaa.MedianBlur(k=(1, 5)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=0.5),
                iaa.Dropout((0.01, 0.3), per_channel=0.5),
                iaa.CoarseDropout((0.03, 0.15), size_percent=(0.03, 0.05), per_channel=0.2)
            ]),
            iaa.OneOf([
                iaa.Sharpen(alpha=(0, 1.0), lightness=(1.0, 1.75)),
                iaa.Emboss(alpha=(0, 1.0), strength=(0.2, 0.75)),
                iaa.EdgeDetect(alpha=(0.1, 0.3)),
                iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
                iaa.Grayscale(alpha=(0.0, 1.0)),
            ])
        ]),
        iaa.OneOf([
            iaa.PiecewiseAffine(scale=(0.01, 0.03)),
            iaa.Affine(scale=(1.0, 1.3),
                       translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                       rotate=(-15, 15),
                       shear=(-15, 15),
                       order=[0, 1],
                       cval=(0, 255),
                       mode=ia.ALL),
            iaa.ElasticTransformation(alpha=(0.01, 2.0), sigma=0.25)
        ]),
    ]), random_order=True)
    augmented_inputs = seq.augment_images(inputs)

    return augmented_inputs
