using JuMP
using CPLEX
using Printf
using LinearAlgebra

function solveMaster(c, A, b, opt_A, opt_B, fea_A, fea_B)
	m_start = Model()
	nx = size(c,2)
	set_optimizer(m_start, CPLEX.Optimizer)
	set_optimizer_attribute(m_start, "CPX_PARAM_SCRIND", 0)
	@variable(m_start, x[i=1:nx] >= 0)
	@constraint(m_start, A*x .<= b)
	if size(fea_A, 1) != 0
		@constraint(m_start, fea_A*x .>= fea_B)
	end
	if size(opt_A, 1) != 0
		@variable(m_start, theta)
		ones_vector = ones(size(opt_A, 1), 1)
		@constraint(m_start, opt_A*x + vec(theta*ones_vector) .>= opt_B)
		@objective(m_start, Min, sum(c[i]*x[i] for i = 1:nx) + theta)
	else
		@objective(m_start, Min, sum(c[i]*x[i] for i = 1:nx))
	end
	optimize!(m_start)
	sol = zeros(1, nx)
	for i = 1:nx
		sol[i] = JuMP.value(x[i])
	end
	sol = vec(sol)
	if size(opt_A, 1) != 0
		return (objective_value(m_start), sol, JuMP.value(theta))
	else
		return (objective_value(m_start), sol, -Inf)
	end
end

function recourseEvaluation(W, h, T, x_t)
	ny = size(W, 2)
	nCstr = size(W, 1)
	m_rec = Model();
	set_optimizer(m_rec, CPLEX.Optimizer);
	set_optimizer_attribute(m_rec, "CPX_PARAM_SCRIND", 0)
	@variable(m_rec, v_plus[1:nCstr] >= 0);#Check size
	@variable(m_rec, v_minus[1:nCstr] >= 0);
	@variable(m_rec, y[1:ny] >= 0);
	@constraint(m_rec,constraint, W*y + v_plus - v_minus - h + T*x_t .<= 0);
	@objective(m_rec, Min, sum(v_plus[i]+v_minus[i] for i=1:nCstr));
	optimize!(m_rec);
	return (objective_value(m_rec), dual.(constraint))
end

function secondStage(q, W, h, T, x_t)
	ny = size(W, 2)
	m_opt = Model()
	set_optimizer(m_opt, CPLEX.Optimizer)
	set_optimizer_attribute(m_opt, "CPX_PARAM_SCRIND", 0)
	@variable(m_opt, y[1:ny] >= 0)
	@constraint(m_opt, opt_cstr, W*y - h + T*x_t .<= 0)
	@objective(m_opt, Min, transpose(q)*y)
	optimize!(m_opt)
	return dual.(opt_cstr)
end

function optimalityCut(q, W, h, T, x_t, theta_t, p)
	nScenario = size(h, 2)
	e1 = 0
	E1 = zeros(1, size(x_t,1))
	for i = 1:nScenario
		pi_k = secondStage(q[i,:], W, h[:,i], T, x_t)
		e1 += p[i]*transpose(pi_k)*h[:,i]
		E1 += p[i]*transpose(pi_k)*T
	end
	w = - sum(E1[i] * x_t[i] for i = 1:size(x_t, 1)) + e1
	if w >= theta_t
		return (e1, E1)
	else
		return (Inf, E1)
	end
end

function L_Shaped_Solver(A, b, c, T, W, h, q, p)
	(val_0, x_0, theta_0) = solveMaster(c, A, b, [], [], [], []) #Get a starting point
	x_t = x_0
	theta_t = theta_0
	val_t = val_0
	val_old = Inf
	nS = size(p, 2)
	opt_A = Float64[]; fea_A = Float64[]
	opt_B = Float64[]; fea_B = Float64[]
	n_iteration = 0
	while val_t != val_old
		n_iteration += 1
		did_feasible = false
		for i = 1:nS
			(rec_val, dual_val) = recourseEvaluation(W, h[:,i], T, x_t)
			if rec_val != 0
				fea_A = cat(fea_A, transpose(dual_val)*T, dims=1)
				fea_B = vcat(fea_B, transpose(dual_val)*h[:,i])
				did_feasible = true
				println("Feasible")
				break;#Not always good
			end
		end
		if !did_feasible
			(subB, subA) = optimalityCut(q, W, h, T, x_t, theta_t, p)
			if subB == Inf
				@printf "-Done in %d iterations\n" n_iteration
				return (val_t, x_t, theta_t)
			end
			opt_A = cat(opt_A, subA, dims=1)
			opt_B = vcat(opt_B, subB)
		end
		val_old = val_t
		(val_t, x_t, theta_t) = solveMaster(c, A, b, opt_A, opt_B, fea_A, fea_B)
	end
	@printf "Done in %d iterations\n" n_iteration
	return (val_t, x_t, theta_t)
end

# Test 1
d = [500 100; 300 300]
q = [-24 -28; -28 -32]
p = [0.4 0.6]

A = [1 1; -1 0; 0 -1]
b = [120; -40; -20]
c = [100 150]

T = [-60 0; 0 -80; 0 0; 0 0]
W = [6 10; 8 5; 1 0; 0 1]
h = [0 0; 0 0; d[1,1] d[2,1]; d[1,2] d[2,2]]

# Test 2
"""
q = [1 1; 1 1; 1 1]
p = [1/3 1/3 1/3]

A = reshape([1], 1, 1)
b = [10]
c = [0]

T = reshape([1; -1], 2, 1)
W = [1 -1; -1 1]
h = [1 2 4; -1 -2 -4]
"""

# Test 3
"""
q = [15 12; 15 12; 15 12; 15 12]
p = [1/4 1/4 1/4 1/4]

A = reshape([0 0], 1, 2)
b = [1]
c = [3 2]

T = [-1 0; 0 -1; 0 0; 0 0; 0 0; 0 0]
W = [3 2; 2 5; 1 0; 0 1; -1 0; 0 -1]
h = [0 0 0 0; 0 0 0 0; 4 4 6 6; 4 8 4 8; (-0.8*4) (-0.8*4) (-0.8*6) (-0.8*6); (-0.8*4) (-0.8*8) (-0.8*4) (-0.8*8)]
"""
(val, sol, theta) = L_Shaped_Solver(A,b,c,T,W,h,q,p)
println("---")
@printf " Objectif : %f\n Theta : %f\n Solution : " val theta
println(sol)
println("---")
