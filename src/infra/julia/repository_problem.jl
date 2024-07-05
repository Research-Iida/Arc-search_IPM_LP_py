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

function extract_elements(qps_data::QPSData, row_type::Int)::Tuple{Vector{Int},Vector{Int},Vector{Float64},Vector{Float64}}
    target_row_number = findall(x -> x == QPSReader.RowType(row_type), qps_data.contypes)
    target_dict_row_idx = Dict{Int,Int}(
        row => idx for (idx, row) in enumerate(target_row_number)
    )
    target_index_sparce_matrix = findall(x -> x ∈ Set(target_row_number), qps_data.arows)

    target_A_rows = [target_dict_row_idx[row_num] for row_num in qps_data.arows[target_index_sparce_matrix]]
    target_A_cols = qps_data.acols[target_index_sparce_matrix]
    target_A_vals = qps_data.avals[target_index_sparce_matrix]

    if row_type == 2
        target_b = qps_data.ucon[target_row_number]
    else
        target_b = qps_data.lcon[target_row_number]
    end

    return tuple(target_A_rows, target_A_cols, target_A_vals, target_b)
end

function load_mps(problem_name::String)::LPOnlyEqualityConstraint
    netlib_path = fetch_netlib()
    qps_data = readqps(netlib_path * "/" * problem_name * ".SIF")

    A_E_rows, A_E_cols, A_E_vals, b_E = extract_elements(qps_data, 1)
    A_L_rows, A_L_cols, A_L_vals, b_L = extract_elements(qps_data, 2)
    A_G_rows, A_G_cols, A_G_vals, b_G = extract_elements(qps_data, 3)

    # julia は1始まりなので python の0始まりに合わせるために -1
    lb_index = findall(x -> !isinf(x), qps_data.lvar) .- 1
    ub_index = findall(x -> !isinf(x), qps_data.uvar) .- 1
    LPOnlyEqualityConstraint(
        qps_data.nvar,
        qps_data.ncon,
        A_E_rows .- 1, A_E_cols .- 1, A_E_vals, b_E,
        A_G_rows .- 1, A_G_cols .- 1, A_G_vals, b_G,
        A_L_rows .- 1, A_L_cols .- 1, A_L_vals, b_L,
        lb_index, qps_data.lvar,
        ub_index, qps_data.uvar,
        qps_data.c
    )
end
