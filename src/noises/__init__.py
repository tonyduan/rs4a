from .noises import *


def parse_noise_from_args(args, device, dim):
    """
    Given a Namespace of arguments, returns the constructed object.
    """
    kwargs = {
        "sigma": args.sigma,
        "lambd": args.lambd,
        "k": args.k,
        "j": args.j,
        "a": args.a
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return eval(args.noise)(device=device, dim=dim, **kwargs)

