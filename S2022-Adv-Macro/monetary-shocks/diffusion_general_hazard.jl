### A Pluto.jl notebook ###
# v0.12.12

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ f620d450-8295-11eb-31ef-7124f66f9ecb
begin
	using LinearAlgebra;
	κ = 0.05; X = 2; 
	dx = 0.01; x = [x for x in -X:dx:X]; 
	Lx = length(x); 
end;

# ╔═╡ de47c07a-82b4-11eb-1546-2b5dcfb1e2e8
md"# Solving the Menu Cost Model in Julia

## Problem
Consider the diffusion equation:

$$\partial_t u(x,t) = -\theta u(x,t) + \kappa \partial_{xx} u(x,t)$$

The main idea here is to use finite difference method and solve it as 

$$\mathbf{D}_t\mathbf{u}_t = \mathbf{A}\mathbf{u}_t \Rightarrow \mathbf{u}_t = e^{\mathbf{A}t}\mathbf{u}_0$$
where 
$\mathbf{A} = \kappa \mathbf{D}_{xx} - \theta \mathbf{I}$.

## Setup
" 

# ╔═╡ a513b5d6-834f-11eb-0261-59c2ed3f9aca
md"## Stationary Distribution

We know that in the stationary distribution

$$0 = \mathbf{A}\hat{\mathbf{u}}$$

so that $\hat{\mathbf{u}}$ is in the null space of $\mathbf{A}$. 
"

# ╔═╡ ae177a18-82ba-11eb-0942-1fee7a19ab16
md"Forming the matrix $\mathbf{A}$ binded to a variable $\theta\in [0,0.5]$:"

# ╔═╡ e82fb22e-8364-11eb-1b7f-f1d9828d22ec
@bind θ̂ html"<input type = 'range' name = 'calvo' min='0' max ='0.5' step='0.01'>
<label for='calvo'>Choose Calvo Parameter</label>
"

# ╔═╡ c8a49a88-82b4-11eb-1577-e3d7979f4b68
begin
	Â = diagm(-1=>ones(Lx-1)*κ/dx^2) + 
	    diagm(0 =>ones(Lx)*((-2*κ)/dx^2-θ̂)) +
	    diagm(1 =>ones(Lx-1)*κ/dx^2);
	midX0 = floor(Int64,Lx/2);
	Â[midX0,1]   = κ/dx^2;
	Â[midX0,end] = κ/dx^2;
	Â[midX0,:] .+= θ̂;
end;

# ╔═╡ fe1bfa44-834f-11eb-3a06-ad4ea61c4ee1
û = nullspace(Â); û = û/sum(û*dx);

# ╔═╡ 121295ca-8351-11eb-3c97-553e155b8080
begin
using Plots; 
plot(x,û,
			color = :blue,
			fillrange = 0, 
			fillalpha = 0.25, 
			legend = :none,
			grid = :none,
			ylim = [0, 1.75],
			title = "Stationary Distribution of Price Gaps",
			xlabel = "Price Gap",
			ylabel = "Density");
end

# ╔═╡ ba207b20-843b-11eb-35ef-a51846905c7a
md"### Comparing with pure Calvo"

# ╔═╡ ca19c1d0-843b-11eb-2812-118883349d28
begin
	Ac = diagm(-1=>ones(Lx-1)*κ/dx^2) + 
	    diagm(0 =>ones(Lx)*((-2*κ)/dx^2-θ̂)) +
	    diagm(1 =>ones(Lx-1)*κ/dx^2);
	midX0c = floor(Int64,Lx/2);
	Ac[midX0c,:] .+= θ̂;
	Ac[2,1] = 0; Ac[end-1,end] = 0;
	Ac[1,1] =  - θ̂;
	Ac[end,end] =  - θ̂;
end

# ╔═╡ 3f18ae88-843c-11eb-039d-1971419b52e5
uc = nullspace(Ac); uc = uc/sum(uc*dx);

# ╔═╡ 5447c320-843c-11eb-1976-f94b7cd40a7b
begin
	plot(x,uc,
		color = :blue,
		fillrange = 0, 
		fillalpha = 0.25, 
		legend = :none,
		grid = :none,
		ylim = [0, 1.75],
		title = "Stationary Distribution of Price Gaps",
		xlabel = "Price Gap",
		ylabel = "Density");
end

# ╔═╡ aa52b9ca-8365-11eb-36d9-ad203ab7415a
md"## Transition Dynamics

Let us know consider some transition dynamics and plot how $u$ converges to the steady state distribution. Starting from a uniform distribution and fixing $\theta$:
"

# ╔═╡ f62c636a-8365-11eb-3924-bd9281a62c05
begin
	u0 = ones(Lx)/(2*X);
	θ  = 0.0;
	A = diagm(-1=>ones(Lx-1)*κ/dx^2) + 
	    diagm(0 =>ones(Lx)*((-2*κ)/dx^2-θ)) +
	    diagm(1 =>ones(Lx-1)*κ/dx^2);
	midX = floor(Int64,Lx/2);
	A[midX,1]   = κ/dx^2;
	A[midX,end] = κ/dx^2;
	A[midX,:] .+= θ;
end;

# ╔═╡ f7cb6aa0-82da-11eb-2887-61a566e5df71
begin
	anim = @animate for τ ∈ 0:0.01:5
		plot(x,exp(A*τ)*u0,
			color = :blue,
			fillrange = 0, 
			fillalpha = 0.25, 
			legend = :none,
			grid = :none,
			ylim = [0, 1.75],
			title = "Stationary Distribution of Price Gaps",
			xlabel = "Price Gap",
			ylabel = "Density");
	end
end

# ╔═╡ 5ff0a386-82db-11eb-1654-f73434df9505
gif(anim, "anim_fps15.gif", fps = 20)

# ╔═╡ 9445c688-82df-11eb-344e-e510a5762c0a
 md"# Monetary Policy Shocks

Let $u(x,0) = \tilde{u}(x-\delta)$ for some $\delta>0$, where $\tilde{u}(x)$ is the stationary distribution. We want to see how the economy converges back to the steady state.
"

# ╔═╡ b8bf54b8-85f3-11eb-3ae5-d1756d8794df
md"### Calvo-Plus Menu Cost Model"

# ╔═╡ 0dd414d2-82e0-11eb-1cf4-276e9523468e
begin 
	δ_ind = 25;
	M     = diagm(δ_ind => ones(Lx-δ_ind));
	u_init = M*û;
	u_init[midX] += (1 - sum(u_init*dx))/dx;
end;

# ╔═╡ 01b7055e-835f-11eb-0428-1b268593f49d
md"Plot the initial distribution:"

# ╔═╡ 6b0be562-8353-11eb-338b-677ff4a325c8
plot(x,u_init,
			color = :blue,
			fillrange = 0, 
			fillalpha = 0.25, 
			legend = :none,
			grid = :none,
			ylim = [0, 1.75],
			title = "Stationary Distribution of Price Gaps",
			xlabel = "Price Gap",
			ylabel = "Density")

# ╔═╡ 9fb67272-82e1-11eb-1a54-2d7605d0db64
begin
	anim_m = @animate for τ ∈ 0:0.02:4
		u = exp(A*τ)*u_init;
		plot(x,u,
			color = :blue,
			fillrange = 0, 
			fillalpha = 0.25, 
			legend = :none,
			grid = :none,
			ylim = [0, 1.75],
			title = "Stationary Distribution of Price Gaps",
			xlabel = "Price Gap",
			ylabel = "Density");
	end
end;

# ╔═╡ aec3f624-82e1-11eb-25a8-2d35181d403a
gif(anim_m, "anim_m_fps20.gif", fps = 15)

# ╔═╡ d26ec9d6-85f3-11eb-1562-ff40afa39813
md"### Calvo"

# ╔═╡ d7619048-85f3-11eb-1bd4-cfac1de62a17
begin 
	u_initc = M*uc;
	u_initc[1] += (1 - sum(u_initc*dx))/dx;
end;

# ╔═╡ 35568e24-85f4-11eb-1411-df0e89fc227a
plot(x,u_initc,
			color = :blue,
			fillrange = 0, 
			fillalpha = 0.25, 
			legend = :none,
			grid = :none,
			ylim = [0, 1.75],
			title = "Stationary Distribution of Price Gaps",
			xlabel = "Price Gap",
			ylabel = "Density")

# ╔═╡ 40e1612e-85f4-11eb-28b7-2d3a0e7a74d6
begin
	anim_mc = @animate for τ ∈ 0:0.05:20
		u = exp(Ac*τ)*u_initc;
		plot(x,u,
			color = :blue,
			fillrange = 0, 
			fillalpha = 0.25, 
			legend = :none,
			grid = :none,
			ylim = [0, 1.75],
			title = "Stationary Distribution of Price Gaps",
			xlabel = "Price Gap",
			ylabel = "Density");
	end
end;

# ╔═╡ 52e891e4-85f4-11eb-2ad2-17b60ba32552
gif(anim_mc, "anim_mc_fps20.gif", fps = 15)

# ╔═╡ Cell order:
# ╟─de47c07a-82b4-11eb-1546-2b5dcfb1e2e8
# ╠═f620d450-8295-11eb-31ef-7124f66f9ecb
# ╟─a513b5d6-834f-11eb-0261-59c2ed3f9aca
# ╟─ae177a18-82ba-11eb-0942-1fee7a19ab16
# ╟─e82fb22e-8364-11eb-1b7f-f1d9828d22ec
# ╠═c8a49a88-82b4-11eb-1577-e3d7979f4b68
# ╠═fe1bfa44-834f-11eb-3a06-ad4ea61c4ee1
# ╠═121295ca-8351-11eb-3c97-553e155b8080
# ╟─ba207b20-843b-11eb-35ef-a51846905c7a
# ╠═ca19c1d0-843b-11eb-2812-118883349d28
# ╠═3f18ae88-843c-11eb-039d-1971419b52e5
# ╠═5447c320-843c-11eb-1976-f94b7cd40a7b
# ╟─aa52b9ca-8365-11eb-36d9-ad203ab7415a
# ╠═f62c636a-8365-11eb-3924-bd9281a62c05
# ╠═f7cb6aa0-82da-11eb-2887-61a566e5df71
# ╠═5ff0a386-82db-11eb-1654-f73434df9505
# ╟─9445c688-82df-11eb-344e-e510a5762c0a
# ╠═b8bf54b8-85f3-11eb-3ae5-d1756d8794df
# ╠═0dd414d2-82e0-11eb-1cf4-276e9523468e
# ╟─01b7055e-835f-11eb-0428-1b268593f49d
# ╠═6b0be562-8353-11eb-338b-677ff4a325c8
# ╠═9fb67272-82e1-11eb-1a54-2d7605d0db64
# ╠═aec3f624-82e1-11eb-25a8-2d35181d403a
# ╟─d26ec9d6-85f3-11eb-1562-ff40afa39813
# ╠═d7619048-85f3-11eb-1bd4-cfac1de62a17
# ╠═35568e24-85f4-11eb-1411-df0e89fc227a
# ╠═40e1612e-85f4-11eb-28b7-2d3a0e7a74d6
# ╠═52e891e4-85f4-11eb-2ad2-17b60ba32552
