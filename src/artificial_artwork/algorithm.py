import attr


@attr.s
class NSTAlgorithm:
    parameters = attr.ib()


@attr.s
class AlogirthmParameters:
    content_image = attr.ib()
    style_image = attr.ib()
    termination_condition = attr.ib()
    output_path = attr.ib()
