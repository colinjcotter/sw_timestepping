import firedrake as fd

class ARK3222:
    def __init__(self, linear, nonlinear, U, dT,
                 lparams, massparams):
        """
        linear - function taking in U and returning the form for linear part
        nonlinear - function taking in U and returning the form for nonlinear part
        U - the solution
        dT = The timestep
        """
        # stage functions
        W = U.function_space()
        Uk2 = fd.Function(W)
        Uk3 = fd.Function(W)

        # some coefficients
        gamma = fd.Constant(1. - 0.5**0.5)
        alpha = fd.Constant((3 + 2*2.**0.5)/6)
        delta = fd.Constant(0.5*0.5**0.5)

        #  Uk1 is just Un for ARK2(2,3,2)
        Uk1 = U
        U0 = U
        U1 = fd.Function(W)
        self.U1 = U1
        self.U0 = U0

        dx = fd.dx
        V = fd.TestFunction(W)
        k2_eqn = (
            fd.inner(V, Uk2 - U0)*dx
            + 2*gamma*dT*nonlinear(V, Uk1)
            + gamma*dT*linear(V, Uk1)
            + gamma*dT*linear(V, Uk2)
            )

        k2prob = fd.NonlinearVariationalProblem(k2_eqn, Uk2)
        self.k2solver = fd.NonlinearVariationalSolver(k2prob,
                                                      options_prefix="k2",
                                                      solver_parameters=lparams)

        k3_eqn = (
            fd.inner(V, Uk3 - U0)*dx
            + (1-alpha)*dT*nonlinear(V, Uk1)
            + alpha*dT*nonlinear(V, Uk2)
            + delta*dT*linear(V, Uk1)
            + delta*dT*linear(V, Uk2)
            + gamma*dT*linear(V, Uk3)
        )

        k3prob = fd.NonlinearVariationalProblem(k3_eqn, Uk3)
        self.k3solver = fd.NonlinearVariationalSolver(k3prob,
                                                      options_prefix="k3",
                                                      solver_parameters=lparams)
        
        unp1_eqn = (
            fd.inner(V, U1 - U0)*dx
            + delta*dT*nonlinear(V, Uk1)
            + delta*dT*nonlinear(V, Uk2)
            + gamma*dT*nonlinear(V, Uk3)
            + delta*dT*linear(V, Uk1)
            + delta*dT*linear(V, Uk2)
            + gamma*dT*linear(V, Uk3)
        )
        
        unp1prob = fd.NonlinearVariationalProblem(unp1_eqn, U1)
        self.unp1solver = fd.NonlinearVariationalSolver(
            unp1prob,
            options_prefix="unp1",
            solver_parameters=massparams)

    def advance(self):
        self.k2solver.solve()
        self.k3solver.solve()
        self.unp1solver.solve()
        self.U0.assign(self.U1)

