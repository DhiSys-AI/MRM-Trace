"""Machine slug — filesystem-safe identifier derived from hostname."""

import re
import socket


def machine_slug() -> str:
    """Return a short, filesystem-safe slug derived from the machine hostname.

    Examples: "dell-g5-5587", "i9-workstation", "ubuntu-vm"
    Override with --machine when the hostname isn't meaningful (cloud VMs).
    """
    name = socket.gethostname().split(".")[0]
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "unknown-host"
