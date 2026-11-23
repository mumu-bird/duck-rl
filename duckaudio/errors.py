class AudioError(Exception):
    """An actionable API error with a stable code."""
    def __init__(self, message, code="audio_error", status=400):
        super().__init__(message)
        self.code = code
        self.status = status

    def as_dict(self):
        return {"error": {"code": self.code, "message": str(self)}}
