using QPSReader

struct LPOnlyEqualityConstraint
    n::Int
    m::Int

    A_E_rows::Vector{Int}
    A_E_cols::Vector{Int}
    A_E_vals::Vector{Float64}
    b_E::Vector{Float64}
    A_G_rows::Vector{Int}
    A_G_cols::Vector{Int}
    A_G_vals::Vector{Float64}
    b_G::Vector{Float64}
    A_L_rows::Vector{Int}
    A_L_cols::Vector{Int}
    A_L_vals::Vector{Float64}
    b_L::Vector{Float64}

    LB_index::Vector{Int}
    LB::Vector{Float64}
    UB_index::Vector{Int}
    UB::Vector{Float64}
    c::Vector{Float64}
end

function load_mps(problem_name::String)::LPOnlyEqualityConstraint
    netlib_path = fetch_netlib()
    qps_data = readqps(netlib_path * "/" * problem_name * ".SIF")

    # julia は1始まりなので python の0始まりに合わせるために -1
    lb_index = findall(x -> !isinf(x), qps_data.lvar) .- 1
    ub_index = findall(x -> !isinf(x), qps_data.uvar) .- 1
    LPOnlyEqualityConstraint(
        qps_data.nvar,
        qps_data.ncon,
        qps_data.arows .- 1,
        qps_data.acols .- 1,
        qps_data.avals,
        qps_data.lcon,
        [], [], [], [],
        [], [], [], [],
        lb_index, qps_data.lvar,
        ub_index, qps_data.uvar,
        qps_data.c
    )
end
