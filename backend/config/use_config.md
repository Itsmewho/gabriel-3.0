Set your envorements and passwords in the .env
Look add the usage in the config and it will guide itself.

- Removed to 2fa login from the frontend. So the legacy code of the:

  # --- HTTP

        HTTPONLY: bool = os.getenv("HTTPONLY", "False").lower() == "true"
        SESSION_EXPIRE_SECONDS = int(os.getenv("SESSION_EXPIRE_SECONDS", "60")) * 60

  Can be ignored or used to build one.
