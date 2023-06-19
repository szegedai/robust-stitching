import numpy as np

def rearrange_activations(activations):
    batch_size = activations.shape[0]
    flat_activations = activations.reshape(batch_size, -1)
    return flat_activations

def cca(x1, x2):
    x1_flat, x2_flat = rearrange_activations(x1), rearrange_activations(x2)
    
    q1, r1 = np.linalg.qr(x1_flat)
    q2, r2 = np.linalg.qr(x2_flat)

    return (np.linalg.norm(q2.T @ q1))**2 / x1_flat.shape[1]


if __name__ == '__main__':
    from sklearn.cross_decomposition import CCA
    import numpy as np

    n = 500000
    features = 10000

    np.random.seed(0)
    U = np.random.random_sample(n).reshape(n//features,features)
    V = np.random.random_sample(n).reshape(n//features,features)

    print(cca(U,V))

    # Sklearn
    sk_cca = CCA(n_components=1)
    sk_cca.fit(U, V)
    U_c, V_c = sk_cca.transform(U, V)
    result = np.corrcoef(U_c.T, V_c.T)[0,1]
    print(result)

    # Matrix
    score = np.corrcoef(U_c.T, V_c.T)
    print(score)