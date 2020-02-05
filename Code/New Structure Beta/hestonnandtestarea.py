
def opti_fun(omega,alpha,beta,gamma_star,h0):
    error = 0
    for n in range(Ntest):
        i=0
        for K in Strikes:
            j=0
            for T in Maturities:
                error += ((BS_formula(prediction(n,i,j))-HNG_Q(alpha, beta, gamma_star, omega, h0, 1, K, r, T, 1))\
                                /BS_formula(prediction(n,i,j)))**2
                j+=1
            i+=1
        n+=1
    return error
    