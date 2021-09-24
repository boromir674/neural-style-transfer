#!/usr/bin/env python


import click

from master import ArtMaster


@click.command()
@click.argument('content_image')
@click.argument('style_image')
@click.option('-i', '--iterations', default=140, show_default=True)
@click.option('-ptm', '--pretrained-model', default='vgg19', show_default=True)
def main(content_image, style_image, iterations, pretrained_model):
    ###### Parameters ######
    content_layer = 'conv4_2'
    style_layers = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)
    ]
    # pretrained_model = 'vgg19'
    #########################

    artm = ArtMaster()
    artm.content_image = content_image
    artm.style_image = style_image
    style_model = artm.build_style(pretrained_model, content_layer, style_layers)
    generated_image = style_model.train(iterations)


if __name__ == '__main__':
    main()
