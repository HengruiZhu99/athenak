"""N12 complete-batch transport controls and their matched comparison modes."""
from typing import NamedTuple


class Transport(NamedTuple):
    send_mib: int = 0
    receive_mib: int = 0
    coalesce: bool = False


TRANSPORT = {
    'cohort': Transport(),
    'locality': Transport(),
    'cohort_cached': Transport(send_mib=96),
    'cohort_packets': Transport(send_mib=96, coalesce=True),
    'cohort_receive_cache': Transport(send_mib=96, receive_mib=96),
    'cohort_cached_packets': Transport(send_mib=96, receive_mib=96, coalesce=True),
}
PACKET_CONTROLS = {
    'cohort': 'spread',
    'cohort_cached': 'cohort',
    'cohort_packets': 'cohort_cached',
    'cohort_receive_cache': 'cohort_cached',
    'cohort_cached_packets': 'cohort_packets',
}


def transport_parameters(mode):
    settings = TRANSPORT.get(mode)
    if settings is None:
        return 'complete_local_batches = false\n'
    text = 'complete_local_batches = true\n'
    if settings.send_mib:
        text += f'send_cache_mib = {settings.send_mib}\n'
    if settings.receive_mib:
        text += f'receive_cache_mib = {settings.receive_mib}\n'
    if settings.coalesce:
        text += 'coalesce_stage_packets = true\n'
    return text
