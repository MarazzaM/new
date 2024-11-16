import tenseal as ts
import numpy as np
from typing import List

def create_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    return context

def encrypt_data(ctx, features, labels):
    encrypted_features = [ts.ckks_vector(ctx, x.tolist()) for x in features]
    encrypted_labels = [ts.ckks_vector(ctx, [y]) for y in labels]
    return encrypted_features, encrypted_labels