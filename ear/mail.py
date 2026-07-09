"""Mail -- native email sending, from the standard library's `smtplib`
and `email` modules alone.

One of the ten basic toolsets (`Toolsets` in memory.md): `email_sender`.
Composing and sending an SMTP message is mechanics, not judgment, so it
ships as a ready BoundTool the same way `ear/web.py`'s tools do -- gated
on a declared SMTP host/port and credential, the credential read as an
environment-variable *name* like every other secret in this package,
never written in memory.md."""

from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

DEFAULT_TIMEOUT = 20.0
DEFAULT_PORT = 587


class MailError(RuntimeError):
    """Sending failed -- no host declared, a bad credential, or the SMTP
    server refused the message. Loud, returned to the model as text."""


@dataclass
class Mail:
    """`send_email`, confined to a declared SMTP host -- there is no
    ambient mail relay this package ever assumes."""

    host: str = ""
    port: int = DEFAULT_PORT
    user_env_var: str = ""
    password_env_var: str = ""
    timeout: float = DEFAULT_TIMEOUT

    def send_email(self, to: str, subject: str, body: str) -> str:
        if not self.host:
            raise MailError("email_sender has no SMTP host declared -- see Toolsets: email_sender in memory.md")
        user = os.environ.get(self.user_env_var) if self.user_env_var else None
        password = os.environ.get(self.password_env_var) if self.password_env_var else None
        message = EmailMessage()
        message["From"] = user or "ear@localhost"
        message["To"] = to
        message["Subject"] = subject
        message.set_content(body)
        try:
            with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as server:
                server.starttls()
                if user and password:
                    server.login(user, password)
                server.send_message(message)
        except (smtplib.SMTPException, OSError) as error:
            raise MailError(f"could not send email to {to!r}: {error}") from error
        return f"sent to {to}"

    def as_tools(self, enabled: set) -> list:
        from .tool_binder import BoundTool

        if "email_sender" not in enabled:
            return []
        return [BoundTool(name="send_email", description="Send an email notification.", handler=self.send_email)]
