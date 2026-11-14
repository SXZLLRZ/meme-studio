class AppError(Exception):
    """Business failure with a message safe to show in the interface."""

    def __init__(
        self,
        code: str,
        user_message: str,
        retryable: bool = False,
        job_id: str | None = None,
        details_ref: str | None = None,
    ) -> None:
        super().__init__(user_message)
        self.code = code
        self.user_message = user_message
        self.retryable = retryable
        self.job_id = job_id
        self.details_ref = details_ref
