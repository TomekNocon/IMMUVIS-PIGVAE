import logging
from collections.abc import Mapping

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all
        processes with their rank prefixed in the log message.

        :param name: The name of the logger. Default is ``__name__``.
        :param rank_zero_only: Whether to force all logs to only occur on the rank zero process. Default is `False`.
        :param extra: (Optional) A dict-like object which provides contextual information. See `logging.LoggerAdapter`.
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(
        self,
        level: int,
        msg: object,
        *args: object,
        exc_info: (
            bool
            | BaseException
            | tuple[type[BaseException], BaseException, object | None]
            | tuple[None, None, None]
            | None
        ) = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message
        with the rank of the process it's being logged from. If `'rank'` is provided,
        then the log will only occur on that rank/process.

        :param level: The level to log at. Look at `logging.__init__.py` for more information.
        :param msg: The message to log.
        :param rank: The rank to log at.
        :param args: Additional args to pass to the underlying logging function.
        :param kwargs: Any additional keyword args to pass to the underlying logging function.
        """
        if self.isEnabledFor(level):
            in_kwargs: dict[str, object] = {}
            if extra is not None:
                in_kwargs["extra"] = dict(extra)
            msg_text, kw = self.process(str(msg), in_kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError(
                    "The `rank_zero_only.rank` needs to be set before use",
                )
            msg_text = rank_prefixed_message(msg_text, current_rank)
            if self.rank_zero_only and current_rank != 0:
                return
            kw.update(
                {
                    "exc_info": exc_info,
                    "stack_info": stack_info,
                    "stacklevel": stacklevel,
                },
            )
            self.logger.log(level, msg_text, *args, **kw)
