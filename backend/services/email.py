import smtplib
import random
from typing import Optional

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from utils.helpers import setup_logger
from config.configure import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS

logger = setup_logger(__name__)


def send_email(
    to_email: str,
    subject: str,
    body: str,
    smtp_host: Optional[str] = None,
    smtp_port: Optional[int] = None,
    smtp_user: Optional[str] = None,
    smtp_pass: Optional[str] = None,
) -> bool:
    """
    Send an email. Returns True if sent, False if failed.
    Params can be overridden for test/dev; otherwise, reads from env.
    """
    try:
        # Prefer arguments, fallback to env vars (for testability)
        smtp_host = SMTP_HOST
        smtp_port_env = SMTP_PORT
        smtp_port = smtp_port if smtp_port is not None else int(smtp_port_env)
        smtp_user = SMTP_USER
        smtp_pass = SMTP_PASS

        if not smtp_host or not smtp_port or not smtp_user or not smtp_pass:
            logger.error(
                "SMTP credentials are incomplete. Check env vars or parameters."
            )
            return False

        message = MIMEMultipart()
        message["From"] = smtp_user
        message["To"] = to_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "html"))

        with smtplib.SMTP_SSL(str(smtp_host), int(smtp_port)) as server:
            server.login(str(smtp_user), str(smtp_pass))
            server.sendmail(str(smtp_user), str(to_email), message.as_string())
            logger.info(f"Email sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Error sending email to {to_email}: {e}")
        return False


def send_2fa_code(email: str) -> Optional[int]:
    """
    Generate and send a 2FA code to the user's email.
    Returns the code if sent, or None if sending failed.
    """
    code = random.randint(100_000, 999_999)  # 6-digit code
    subject = "Your 2FA Code"
    body = f"""
    <html>
        <body>
            <p>Your 2FA code is:</p>
            <h1>{code}</h1>
            <p>Please enter this code to complete your login process. This code will expire in 5 minutes.</p>
        </body>
    </html>
    """
    sent = send_email(email, subject, body)
    return code if sent else None
