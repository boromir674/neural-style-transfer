from typing import List, Callable
from numpy.typing import NDArray


class ImageProcessor:
    def process(self, image: NDArray, pipeline: List[Callable[[NDArray], NDArray]]) -> NDArray:
        if len(pipeline) > 0:
            processor = pipeline[0]
            pipeline = pipeline[1:]
            return self.process(processor(image), pipeline)
        return image
