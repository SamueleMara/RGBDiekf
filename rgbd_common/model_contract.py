"""Fail explicitly where the supplied variable-model objective is undefined."""


class UnspecifiedModelSelection(ValueError):
    pass


def require_model_selection(config):
    if callable(getattr(config,'structure_prior',None)) and callable(getattr(config,'initializer_log_prior',None)):
        return
    raise UnspecifiedModelSelection(
        'UNSPECIFIED_MODEL_SELECTION: theory v2 M6, A7, D7 and implementation '
        'contract Sections 5/10 require a comparable variable-point/group score, '
        'but do not define the normalized p(G,A), point-count law or '
        'cross-dimensional trajectory reference measure. Approved P0/Q/R do '
        'not supply them. The former p(B)=1/[B(B+1)], N*log(B) and '
        'p(N)=1/[(N+1)(N+2)] were implementation inventions and have been '
        'removed from production. No invented group penalty or bootstrap '
        'segmentation is used. The conditional IEKF/UKF components remain '
        'testable, but the full reconstruction cannot be called defined or accepted.'
    )
