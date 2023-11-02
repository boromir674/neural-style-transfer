from typing import List

import tensorflow as tf
from software_patterns import Proxy


class TensorflowSessionRunnerSubject:
    def __init__(self, interactive_session) -> None:
        self.interactive_session = interactive_session

    def run(self, *args, **kwargs):
        return self.interactive_session.run(*args, **kwargs)


class TensorflowSessionRunner(Proxy):
    def __init__(self, real_subject) -> None:
        super().__init__(real_subject)
        # self._proxy_subject IS a reference to an
        # TensorflowSessionRun  nerSubject instance
        self.args_history: List[str] = []

    def run(self, *args, **kwargs):
        """# Using the `close()` method.
        sess = tf.compat.v1.Session()
        sess.run(...)
        sess.close()

        OR

        # Using the context manager.
        with tf.compat.v1.Session() as sess:
        sess.run(...)
        """
        session_run_callable = self._proxy_subject.run
        args_str = f"[{', '.join((str(_) for _ in args))}]"
        kwargs_str = f"[{', '.join((f'{k}={v}' for k, v in kwargs.items()))}]"
        self.args_history.append(f"ARGS: {args_str}, KWARGS: {kwargs_str}")
        try:
            return session_run_callable(*args, **kwargs)
        except Exception as tensorflow_error:
            raise TensorflowSessionRunError(
                "Tensorflow error occured, when"
                f"running session with input args {args_str} and kwargs {kwargs_str}"
            ) from tensorflow_error

    @property
    def session(self):
        return self._proxy_subject.interactive_session

    @classmethod
    def with_default_graph_reset(cls):
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()
        return TensorflowSessionRunner(
            TensorflowSessionRunnerSubject(tf.compat.v1.InteractiveSession())
        )


class TensorflowSessionRunError(Exception):
    pass
