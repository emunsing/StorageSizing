import numpy
import scipy.sparse as sps
import sys

def createABMatrices( Length, delta_t, eff_in, eff_out, P_min, P_max, E_min, E_max):
    # Use a linear program to solve the following problem: 

    #  minimize   h*store_price + sum(price_i * (charge_i + discharge_i) )
    #   h,e,c,d

    # Over the control variables: 
    # e (Charge level, kWh), c (charge power kW), d (discharge power kW),
    # h (storage size, kWh)
    # Sign convention: Power sold to grid is negative, 
    #                  power bought from grid is positive

    # Subject to:
    # Equality constraints:
    # e_0 = 0
    # e_i - e_(i-1) - (n_in * c_i)*delta_T + (1/n_out * d_i)*delta_t = 0
    #
    # Inequality constraints:
    # e_min * h <= e_i <= e_max * h     for all i
    #   0   <= c_i <= p_max     for all i
    # p_min <= d_i <= 0     for all i
    # OR:
    # h * -e_max + e_i <= 0
    # h * e_min  - e_i <= 0
    # c_i <= p_max
    # -1* c_i <= 0
    # d_i <= 0
    # -1* d_i <= -1* p_min

    # Prepping for matlab:
    # minimize f'*x
    #    x
    # s.t. A*x <= b 
    #      A_eq = b_eq

    # Control variable x:
    # X: [h, e_1....e_N, c_1...c_N, d_1...d_N]

    print('Length = %s' % Length)
    sys.stdout.flush()

    E_init = 0
    
    ## Cost Matrix and Function

    # X: [h, e_1....e_N, c_1...c_N, d_1...d_N]
    # Cannot have a 1xn array (e.g. sps.coo_matrix) when passed to scipy.optimize.linprog:
    #    that function uses len(c) which is ambiguous. Would be better if this had shape(c)[0]
    #C = np.concatenate([[store_price],[0]*(Length+1),c_grid,c_grid],axis=1)  # No cost for storage state; charged for what we consume, get paid for what we discharge
    
    ## Equality Constraints
    # Control variables:
    # Length of one variable for each time step

    #  E(0) = E_init
    #  E(k+1) - E(k) + nT*Ps(k) = 0

    # [E(0) -----E(N) |C(1)...C(N) | D(1)...D(N)]

    # A_eq = [A_eq0, A_eq1 ,  A_eq_2   ,   A_eq3    ]
    # [0  1  0  0  0  0  0   0  0  0   0   0   0   0;
    #  0 -1  1  0  0  0 -nT  0  0  0  -1/nT 0  0   0;
    #  0  0 -1  1  0  0  0 -nT  0  0   0 -1/nT 0   0;
    #  0  0  0 -1  1  0  0  0  -nT 0   0   0 -1/nT 0;
    #  0  0  0  0 -1  1  0  0  0  -nT  0   0   0 -1/nT];

    # A_eq = [A_eq0, A_eq1, A_eq_2, A_eq3]

    A_eq_0 = sps.coo_matrix((Length+1,1))

    A_eq_1 = sps.eye(Length+1) - sps.eye(Length+1,k=-1) # Identity - off-diagonal term

    A_eq_2 = sps.vstack( [sps.coo_matrix((1,Length)), 
                          -1* eff_in * delta_t * sps.eye(Length)])

    A_eq_3 = sps.vstack( [sps.coo_matrix((1,Length)),
                          -1* delta_t / eff_out * sps.eye(Length)])
    
    A_eq = sps.hstack( [A_eq_0, A_eq_1, A_eq_2, A_eq_3])

    B_eq = sps.vstack( [ E_init, sps.coo_matrix((Length,1)) ])

    # Use the following section to force h to be a specific size:
    #h_constant = sps.hstack( [1, sps.coo_matrix((1, Length*3+1))] );
    #A_eq = sps.vstack( [h_constant, A_eq ] )
    #B_eq = sps.vstack( [1, B_eq] )


    ## Inequality Constraints
    # A*x = b
    # Each block must have width Length*3+2 and height Length+1
    # A = [E_1  ; E_2  ; C_1 ;  C_2  ; D_1  ; D_2]
    # OR
    # A = [E_1  ; -1* E_1; C_1; -1*C_1; D_1  ; -1*D_1]
    # b = [E_max;  E_min; C_max; C_min; D_max; D_min]

    # H positivity constraints
    # -h <=0
    H_0 = sps.hstack([ -1, sps.coo_matrix((1, Length*3+1)) ] )
    B_0 = 0

    # E_max Constraints
    # h * -e_max + e_i <= 0
    # h * e_min  - e_i <= 0
    E_1 = sps.hstack( [-E_max * numpy.ones((Length+1, 1)), 
                       sps.eye(Length+1),
                       sps.coo_matrix((Length+1,Length*2)) ])
    #  [   h      E_i     C_i    D_i  ]
    #  [-e_max  1 0 0 0  0 0 0  0 0 0 ;
    #   -e_max  0 1 0 0  0 0 0  0 0 0 ;
    #   -e_max  0 0 1 0  0 0 0  0 0 0 ;
    #   -e_max  0 0 0 1  0 0 0  0 0 0 ];
    B_1 = sps.coo_matrix((Length+1,1))

    # E_min Constraints
    E_2 = sps.hstack( [ E_min * numpy.ones((Length+1, 1)),
                       -1 * sps.eye(Length+1),
                       sps.coo_matrix((Length+1,Length*2)) ])
    B_2 = sps.coo_matrix((Length+1,1))

    # Charge Power Constraints
    # 0 <= c_i <= pmax     for all i
    # C max Constraints
    C_1 = sps.hstack( [ sps.coo_matrix((Length,Length+2)), sps.eye(Length), sps.coo_matrix((Length,Length)) ] )
    #  [h   E_i     C_i    D_i  ]
    #   0  0 0 0 0  1 0 0  0 0 0 ;
    #   0  0 0 0 0  0 1 0  0 0 0 ;
    #   0  0 0 0 0  0 0 1  0 0 0 ];
    B_3 = P_max * numpy.ones((Length,1))

    #C Min Constraints
    C_2 = -1 * C_1
    B_4 = sps.coo_matrix((Length,1))


    # Discharge Power constraints
    #  pmin <= d_i <= 0     for all i
    # D max Constraints
    D_1 = sps.hstack( [sps.coo_matrix((Length,Length*2 +2)), sps.eye(Length) ])
    #  [h    E_i     C_i    D_i  ]
    #   0  0 0 0 0  0 0 0  1 0 0 ;
    #   0  0 0 0 0  0 0 0  0 1 0 ;
    #   0  0 0 0 0  0 0 0  0 0 1 ];
    B_5 = sps.coo_matrix((Length,1))

    #D Min Constraints
    D_2 = -1 * D_1
    B_6 = -1* P_min * numpy.ones((Length,1))

    #Combining Matricies
    A = sps.vstack( [H_0, E_1 , E_2 , C_1 , C_2 , D_1 , D_2] )
    b = sps.vstack( [B_0, B_1 , B_2 , B_3 , B_4 , B_5 , B_6] )

    #disp('Inequality Constraints Complete; Running linprog...');
    
    #print('Size of C    (aka c): %s ' % C.shape)
    print('Size of A    (aka G): %s x %s' % A.shape)
    print('Size of b    (aka h): %s x %s' % b.shape)
    print('Size of A_eq (aka A): %s x %s' % A_eq.shape)
    print('Size of B_eq (aka b): %s x %s' % B_eq.shape)
    sys.stdout.flush()
    
    return (A, b, A_eq, B_eq)

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return numpy.isnan(y), lambda z: z.nonzero()[0]