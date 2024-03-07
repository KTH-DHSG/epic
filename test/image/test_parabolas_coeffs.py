import sympy as sp


def test_coeffs():
    x = sp.Symbol('x')
    y = sp.Symbol('y')

    m0 = sp.Symbol('m0')
    m1 = sp.Symbol('m1')
    m2 = sp.Symbol('m2')
    m3 = sp.Symbol('m3')
    m4 = sp.Symbol('m4')
    m5 = sp.Symbol('m5')

    k0 = sp.Symbol('k0')
    k1 = sp.Symbol('k1')
    k2 = sp.Symbol('k2')
    k3 = sp.Symbol('k3')
    k4 = sp.Symbol('k4')
    k5 = sp.Symbol('k5')

    p1 = m0 * x**2 + m1 * x * y + m2 * y**2 + m3 * x + m4 * y + m5
    p2 = k0 * x**2 + k1 * x * y + k2 * y**2 + k3 * x + k4 * y + k5

    sol_x_p1 = sp.solve(p1, x)
    p2_with_x1 = p2.subs(x, sol_x_p1[0])
    p2_with_x2 = p2.subs(x, sol_x_p1[1])
    p2_mult = sp.simplify(p2_with_x1 * p2_with_x2)
    p2_poly = sp.Poly(p2_mult, y)
    all_coeffs = p2_poly.all_coeffs()

    # Coefficients of the polynomial
    print("Coefficients for x as function of y:")
    print("  -> solution x1: ", sol_x_p1[0])
    print("  -> solution x2: ", sol_x_p1[1])

    print("\nCoefficients for y:")
    # c1 * y**4 + c2 * y**3 + c3 * y**2 + c4 * y + c5 = 0
    print("  -> c1: ", all_coeffs[0])
    print("  -> c2: ", all_coeffs[1])
    print("  -> c3: ", all_coeffs[2])
    print("  -> c4: ", all_coeffs[3])
    print("  -> c5: ", all_coeffs[4])
    # get solution by calculating np.roots([c1, c2, c3, c4, c5])


if __name__ == '__main__':
    test_coeffs()
